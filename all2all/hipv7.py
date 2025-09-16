import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# HIP kernel compilation setup
os.environ["CXX"] = "clang++"

CPP_WRAPPER = """
// --- V6 Kernels (some are reused) ---
void hip_count_and_map_dispatches(torch::Tensor indices, torch::Tensor send_counts, torch::Tensor dispatch_map,
                                  int num_tokens, int experts_per_token, int num_local_experts);
void hip_unpack_and_reorganize(torch::Tensor recv_buf, torch::Tensor expert_x, torch::Tensor expert_meta,
                                torch::Tensor expert_num_tokens, torch::Tensor recv_data_byte_offsets,
                                torch::Tensor recv_meta_byte_offsets, torch::Tensor recv_counts,
                                int total_recv, int num_local_experts, int hidden_dim, int max_recv);
void hip_count_backward_sends(torch::Tensor expert_meta, torch::Tensor expert_num_tokens,
                              torch::Tensor send_counts, int num_local_experts, int max_recv);
void hip_gather_and_pack(torch::Tensor expert_y, torch::Tensor expert_meta, torch::Tensor expert_num_tokens,
                          torch::Tensor temp_send_counts, torch::Tensor send_buf,
                          torch::Tensor send_data_offsets, torch::Tensor send_meta_offsets,
                          int num_local_experts, int max_recv, int hidden_dim);
void hip_unpack_to_intermediate(torch::Tensor recv_buf, torch::Tensor intermediate_buf,
                                torch::Tensor recv_data_byte_offsets, torch::Tensor recv_meta_byte_offsets,
                                torch::Tensor recv_counts, int total_recv, int hidden_dim, int experts_per_token);
void hip_final_reduction_smem(torch::Tensor intermediate_buf, torch::Tensor weights, torch::Tensor out_tokens,
                              int num_tokens, int hidden_dim, int experts_per_token);

// --- V7 New/Modified Kernels ---
void hip_permute_and_pack_from_sorted_map(torch::Tensor original_tokens, torch::Tensor indices,
                                          torch::Tensor sorted_dispatch_indices, torch::Tensor sorted_dispatch_map_flat,
                                          torch::Tensor send_buf, torch::Tensor send_offsets,
                                          torch::Tensor send_data_offsets, torch::Tensor send_meta_offsets,
                                          int num_tokens, int experts_per_token, int hidden_dim, int rank, int num_local_experts);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

constexpr const int BLOCK_SIZE = 256; // Increased block size for potentially better occupancy
constexpr const int META_DIM = 5;
constexpr const int WORLD_SIZE = 8;

#define VECTORIZE_COPY(dst, src, hidden_dim) \\
    do { \\
        const int4* src_vec = reinterpret_cast<const int4*>(src); \\
        int4* dst_vec = reinterpret_cast<int4*>(dst); \\
        _Pragma("unroll") for (int h = 0; h < (hidden_dim) / 8; ++h) { \\
            dst_vec[h] = src_vec[h]; \\
        } \\
    } while (0)

// --- V6 Kernels (Unchanged, included for completeness) ---

__global__ void count_and_map_kernel(const int* indices, int* send_counts, int* dispatch_map,
                                     int num_tokens, int experts_per_token, int num_local_experts) {
    __shared__ int s_send_counts[WORLD_SIZE];
    if (threadIdx.x < WORLD_SIZE) { s_send_counts[threadIdx.x] = 0; }
    __syncthreads();
    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < num_tokens * experts_per_token; tid += gridDim.x * blockDim.x) {
        int token_idx = tid / experts_per_token;
        int expert_idx = tid % experts_per_token;
        int expert_id = indices[token_idx * experts_per_token + expert_idx];
        int dst_rank = expert_id / num_local_experts;
        atomicAdd(&s_send_counts[dst_rank], 1);
        int map_offset = tid * 3;
        dispatch_map[map_offset + 0] = dst_rank;
        dispatch_map[map_offset + 1] = token_idx;
        dispatch_map[map_offset + 2] = expert_idx;
    }
    __syncthreads();
    if (threadIdx.x < WORLD_SIZE) { atomicAdd(&send_counts[threadIdx.x], s_send_counts[threadIdx.x]); }
}

__global__ void unpack_and_reorganize_kernel(const unsigned char* recv_buf, __half* expert_x, int* expert_meta,
                                             int* expert_num_tokens, const int* recv_data_byte_offsets,
                                             const int* recv_meta_byte_offsets, const int* recv_counts,
                                             int total_recv, int num_local_experts, int hidden_dim, int max_recv) {
    __shared__ int s_recv_offsets[WORLD_SIZE + 1];
    if (threadIdx.x == 0) {
        s_recv_offsets[0] = 0;
        for (int i = 0; i < WORLD_SIZE; ++i) s_recv_offsets[i+1] = s_recv_offsets[i] + recv_counts[i];
    }
    __syncthreads();
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_recv) return;
    int src_rank = 0;
    _Pragma("unroll") for (int i = 1; i < WORLD_SIZE + 1; ++i) if (tid >= s_recv_offsets[i]) src_rank = i;
    int intra_rank_idx = tid - s_recv_offsets[src_rank];
    const int* meta_read_ptr = reinterpret_cast<const int*>(recv_buf + recv_meta_byte_offsets[src_rank] + intra_rank_idx * META_DIM * sizeof(int));
    int global_eid = meta_read_ptr[0];
    int local_eid = global_eid % num_local_experts;
    int pos = atomicAdd(&expert_num_tokens[local_eid], 1);
    if (pos < max_recv) {
        const __half* data_read_ptr = reinterpret_cast<const __half*>(recv_buf + recv_data_byte_offsets[src_rank] + intra_rank_idx * hidden_dim * sizeof(__half));
        __half* data_write_ptr = &expert_x[local_eid * max_recv * hidden_dim + pos * hidden_dim];
        VECTORIZE_COPY(data_write_ptr, data_read_ptr, hidden_dim);
        int* meta_write_ptr = &expert_meta[local_eid * max_recv * META_DIM + pos * META_DIM];
        for (int m = 0; m < META_DIM; ++m) meta_write_ptr[m] = meta_read_ptr[m];
    }
}

__global__ void count_backward_kernel(const int* expert_meta, const int* expert_num_tokens, int* send_counts, int num_local_experts, int max_recv) {
    int local_eid = blockIdx.x;
    int num_tokens_in_expert = expert_num_tokens[local_eid];
    for (int i = threadIdx.x; i < num_tokens_in_expert; i += blockDim.x) {
        if (i < max_recv) {
            int dst_rank = expert_meta[local_eid * max_recv * META_DIM + i * META_DIM + 1];
            atomicAdd(&send_counts[dst_rank], 1);
        }
    }
}

__global__ void gather_and_pack_kernel(const __half* expert_y, const int* expert_meta, const int* expert_num_tokens, int* temp_send_counts, unsigned char* send_buf, const int* send_data_offsets, const int* send_meta_offsets, int num_local_experts, int max_recv, int hidden_dim) {
    int local_eid = blockIdx.x;
    int token_idx_in_expert = threadIdx.x;
    int num_valid_tokens = expert_num_tokens[local_eid];
    if (local_eid >= num_local_experts || token_idx_in_expert >= num_valid_tokens) return;
    const int* meta_read_ptr = &expert_meta[local_eid * max_recv * META_DIM + token_idx_in_expert * META_DIM];
    int dst_rank = meta_read_ptr[1];
    int pos = atomicAdd(&temp_send_counts[dst_rank], 1);
    const __half* data_read_ptr = &expert_y[local_eid * max_recv * hidden_dim + token_idx_in_expert * hidden_dim];
    __half* data_write_ptr = reinterpret_cast<__half*>(send_buf + send_data_offsets[dst_rank] + pos * hidden_dim * sizeof(__half));
    VECTORIZE_COPY(data_write_ptr, data_read_ptr, hidden_dim);
    int* meta_write_ptr = reinterpret_cast<int*>(send_buf + send_meta_offsets[dst_rank] + pos * META_DIM * sizeof(int));
    for (int m = 0; m < META_DIM; ++m) meta_write_ptr[m] = meta_read_ptr[m];
}

__global__ void unpack_to_intermediate_kernel(const unsigned char* recv_buf, __half* intermediate_buf, const int* recv_data_byte_offsets, const int* recv_meta_byte_offsets, const int* recv_counts, int total_recv, int hidden_dim, int experts_per_token) {
    __shared__ int s_recv_offsets[WORLD_SIZE + 1];
    if (threadIdx.x == 0) {
        s_recv_offsets[0] = 0;
        for (int i = 0; i < WORLD_SIZE; ++i) s_recv_offsets[i+1] = s_recv_offsets[i] + recv_counts[i];
    }
    __syncthreads();
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_recv) return;
    int src_rank = 0;
    _Pragma("unroll") for (int i = 1; i < WORLD_SIZE + 1; ++i) if (tid >= s_recv_offsets[i]) src_rank = i;
    int intra_rank_idx = tid - s_recv_offsets[src_rank];
    const int* meta_read_ptr = reinterpret_cast<const int*>(recv_buf + recv_meta_byte_offsets[src_rank] + intra_rank_idx * META_DIM * sizeof(int));
    const __half* data_read_ptr = reinterpret_cast<const __half*>(recv_buf + recv_data_byte_offsets[src_rank] + intra_rank_idx * hidden_dim * sizeof(__half));
    int src_token = meta_read_ptr[2];
    int src_k = meta_read_ptr[3];
    __half* intermediate_write_ptr = &intermediate_buf[(src_token * experts_per_token + src_k) * hidden_dim];
    VECTORIZE_COPY(intermediate_write_ptr, data_read_ptr, hidden_dim);
}

__global__ void final_reduction_kernel_smem(const __half* intermediate_buf, const float* weights, __half* out_tokens, int num_tokens, int hidden_dim, int experts_per_token) {
    extern __shared__ float s_weights[];
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    if (threadIdx.x < experts_per_token) s_weights[threadIdx.x] = weights[token_idx * experts_per_token + threadIdx.x];
    __syncthreads();
    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float sum = 0.0f;
        _Pragma("unroll") for (int k = 0; k < experts_per_token; ++k) {
            sum += s_weights[k] * __half2float(intermediate_buf[(token_idx * experts_per_token + k) * hidden_dim + h]);
        }
        out_tokens[token_idx * hidden_dim + h] = __float2half(sum);
    }
}

// --- V7 New Kernel: Permute from sorted map (NO ATOMICS) ---
__global__ void permute_and_pack_from_sorted_map_kernel(
    const __half* original_tokens, const int* indices,
    const long* sorted_dispatch_indices, const int* sorted_dispatch_map_flat,
    unsigned char* send_buf, const int* send_offsets,
    const int* send_data_offsets, const int* send_meta_offsets,
    int num_total_dispatches, int hidden_dim, int rank, int num_local_experts, int experts_per_token)
{
    // Each thread handles one dispatch item from the sorted list
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_total_dispatches) return;

    // 1. Find which rank this thread's item belongs to and its intra-rank index.
    // send_offsets contains the starting index for each rank's data in the sorted map.
    int dst_rank = sorted_dispatch_map_flat[tid * 3 + 0];
    int intra_rank_idx = tid - send_offsets[dst_rank];

    // 2. Get the original token_idx and expert_idx for this dispatch item.
    // sorted_dispatch_map_flat is already sorted by dst_rank.
    int token_idx  = sorted_dispatch_map_flat[tid * 3 + 1];
    int expert_idx = sorted_dispatch_map_flat[tid * 3 + 2];

    // 3. Write data and metadata to the correct position WITHOUT atomics.
    // The position 'pos' is simply the intra_rank_idx.
    __half* data_write_ptr = reinterpret_cast<__half*>(send_buf + send_data_offsets[dst_rank] + intra_rank_idx * hidden_dim * sizeof(__half));
    const __half* data_read_ptr = &original_tokens[token_idx * hidden_dim];
    VECTORIZE_COPY(data_write_ptr, data_read_ptr, hidden_dim);

    int* meta_write_ptr = reinterpret_cast<int*>(send_buf + send_meta_offsets[dst_rank] + intra_rank_idx * META_DIM * sizeof(int));
    int expert_id = indices[token_idx * experts_per_token + expert_idx];
    meta_write_ptr[0] = expert_id;
    meta_write_ptr[1] = rank;
    meta_write_ptr[2] = token_idx;
    meta_write_ptr[3] = expert_idx;
    meta_write_ptr[4] = 0;
}


// --- C++ Wrapper Functions ---
void hip_count_and_map_dispatches(torch::Tensor i, torch::Tensor sc, torch::Tensor dm, int n, int e, int l) { count_and_map_kernel<<<(n*e+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(i.data_ptr<int>(), sc.data_ptr<int>(), dm.data_ptr<int>(), n, e, l); }
void hip_unpack_and_reorganize(torch::Tensor r, torch::Tensor ex, torch::Tensor em, torch::Tensor en, torch::Tensor rdo, torch::Tensor rmo, torch::Tensor rc, int t, int l, int h, int mr) { unpack_and_reorganize_kernel<<<(t+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>((const unsigned char*)r.data_ptr(), (__half*)ex.data_ptr<at::Half>(), em.data_ptr<int>(), en.data_ptr<int>(), rdo.data_ptr<int>(), rmo.data_ptr<int>(), rc.data_ptr<int>(), t, l, h, mr); }
void hip_count_backward_sends(torch::Tensor em, torch::Tensor en, torch::Tensor sc, int l, int mr) { count_backward_kernel<<<l, BLOCK_SIZE>>>(em.data_ptr<int>(), en.data_ptr<int>(), sc.data_ptr<int>(), l, mr); }
void hip_gather_and_pack(torch::Tensor ey, torch::Tensor em, torch::Tensor en, torch::Tensor t, torch::Tensor s, torch::Tensor sdo, torch::Tensor smo, int l, int mr, int h) { gather_and_pack_kernel<<<l, BLOCK_SIZE>>>((const __half*)ey.data_ptr<at::Half>(), em.data_ptr<int>(), en.data_ptr<int>(), t.data_ptr<int>(), (unsigned char*)s.data_ptr(), sdo.data_ptr<int>(), smo.data_ptr<int>(), l, mr, h); }
void hip_unpack_to_intermediate(torch::Tensor rb, torch::Tensor ib, torch::Tensor rdo, torch::Tensor rmo, torch::Tensor rc, int t, int h, int e) { unpack_to_intermediate_kernel<<<(t+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>((const unsigned char*)rb.data_ptr(), (__half*)ib.data_ptr<at::Half>(), rdo.data_ptr<int>(), rmo.data_ptr<int>(), rc.data_ptr<int>(), t, h, e); }
void hip_final_reduction_smem(torch::Tensor ib, torch::Tensor w, torch::Tensor ot, int n, int h, int e) { size_t smem = e * sizeof(float); final_reduction_kernel_smem<<<n, BLOCK_SIZE, smem>>>((const __half*)ib.data_ptr<at::Half>(), w.data_ptr<float>(), (__half*)ot.data_ptr<at::Half>(), n, h, e); }

void hip_permute_and_pack_from_sorted_map(torch::Tensor o, torch::Tensor i, torch::Tensor sdi, torch::Tensor sdmf, torch::Tensor s, torch::Tensor so, torch::Tensor sdo, torch::Tensor smo, int n, int e, int h, int r, int l) {
    int total_dispatches = n * e;
    permute_and_pack_from_sorted_map_kernel<<<(total_dispatches + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        (const __half*)o.data_ptr<at::Half>(), i.data_ptr<int>(),
        sdi.data_ptr<long>(), sdmf.data_ptr<int>(),
        (unsigned char*)s.data_ptr(), so.data_ptr<int>(),
        sdo.data_ptr<int>(), smo.data_ptr<int>(),
        total_dispatches, h, r, l, e
    );
}

"""

# Compile HIP module
hip_module = load_inline(
    name='hip_all2all_v7_sort', # Renamed to avoid cache conflicts
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=[
        'hip_count_and_map_dispatches', 'hip_unpack_and_reorganize',
        'hip_count_backward_sends', 'hip_gather_and_pack',
        'hip_unpack_to_intermediate', 'hip_final_reduction_smem',
        'hip_permute_and_pack_from_sorted_map' # New kernel
    ],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3"],
)

class HIPAllToAll:
    META_DIM = 5

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg
        num_tokens, experts_per_token = indices.shape
        total_dispatches = num_tokens * experts_per_token
        indices_int = indices.to(torch.int32)

        # 1. Count tokens per destination and create the initial dispatch map (same as v6)
        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        dispatch_map = torch.empty(total_dispatches, 3, dtype=torch.int32, device=device)
        hip_module.hip_count_and_map_dispatches(indices_int, send_counts, dispatch_map, num_tokens, experts_per_token, self.num_local_experts)

        # --- NEW: Sort the dispatch map by destination rank ---
        # This is the core of the new strategy.
        # We sort based on the first column (dst_rank) of the dispatch_map.
        # `sorted_indices` will store the original positions, which we don't need for permutation,
        # but the sorting operation gives us the sorted map.
        # Note: The sort operation itself happens on the GPU and is highly optimized.
        _, sorted_indices = torch.sort(dispatch_map[:, 0])
        sorted_dispatch_map = dispatch_map[sorted_indices]

        # 2. Exchange counts (same as v6, needed for All-to-All)
        recv_counts_t = torch.empty_like(send_counts, dtype=torch.long)
        dist.all_to_all_single(recv_counts_t, send_counts.to(torch.long))
        recv_counts = recv_counts_t.to(torch.int32)
        total_recv = int(recv_counts.sum().item())

        # 3. Prepare buffers for fused communication (same as v6)
        send_data_bytes = send_counts * cfg.hidden_dim * 2
        send_meta_bytes = send_counts * self.META_DIM * 4
        send_split_bytes = send_data_bytes + send_meta_bytes
        send_buf_total_bytes = send_split_bytes.sum().item()
        send_buf = torch.empty(send_buf_total_bytes, dtype=torch.uint8, device=device)

        # Calculate offsets for data and metadata within the single send_buf
        send_block_offsets = torch.cumsum(send_split_bytes, 0, dtype=torch.int32) - send_split_bytes
        send_data_offsets = send_block_offsets
        send_meta_offsets = send_block_offsets + send_data_bytes

        # Calculate offsets needed by the new kernel to know where each rank's data begins in the sorted map
        send_offsets_in_map = torch.cumsum(send_counts, 0, dtype=torch.int32) - send_counts

        # 4. --- MODIFIED: Call the new permutation kernel ---
        # This kernel reads from the `sorted_dispatch_map` and writes to `send_buf` without any atomics.
        hip_module.hip_permute_and_pack_from_sorted_map(
            dp_x, indices_int, sorted_indices, sorted_dispatch_map.view(-1),
            send_buf, send_offsets_in_map,
            send_data_offsets, send_meta_offsets,
            num_tokens, experts_per_token, cfg.hidden_dim, self.rank, self.num_local_experts
        )

        # 5. Perform All-to-All and unpack (same as v6)
        recv_data_bytes = recv_counts * cfg.hidden_dim * 2
        recv_meta_bytes = recv_counts * self.META_DIM * 4
        recv_split_bytes = recv_data_bytes + recv_meta_bytes
        recv_buf = torch.empty(recv_split_bytes.sum().item(), dtype=torch.uint8, device=device)
        dist.all_to_all_single(recv_buf, send_buf, recv_split_bytes.tolist(), send_split_bytes.tolist())

        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM), dtype=torch.int32, device=device)

        if total_recv > 0:
            recv_block_offsets = torch.cumsum(recv_split_bytes, 0, dtype=torch.int32) - recv_split_bytes
            recv_data_byte_offsets = recv_block_offsets
            recv_meta_byte_offsets = recv_block_offsets + recv_data_bytes
            hip_module.hip_unpack_and_reorganize(recv_buf, expert_x, expert_meta, expert_num_tokens, recv_data_byte_offsets, recv_meta_byte_offsets, recv_counts, total_recv, self.num_local_experts, cfg.hidden_dim, self.max_recv)

        return expert_num_tokens, expert_x, expert_meta

    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, expert_meta: torch.Tensor, expert_y: torch.Tensor, expert_num_tokens: torch.Tensor, num_tokens: int):
        # The combine logic from v6 is already highly optimized with the two-stage approach.
        # We can reuse it directly.
        device = out_tokens.device
        cfg = self.cfg

        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        hip_module.hip_count_backward_sends(expert_meta, expert_num_tokens, send_counts, self.num_local_experts, self.max_recv)

        recv_counts_t = torch.empty_like(send_counts, dtype=torch.long)
        dist.all_to_all_single(recv_counts_t, send_counts.to(torch.long))
        recv_counts = recv_counts_t.to(torch.int32)
        total_recv = int(recv_counts.sum().item())

        total_sends = int(send_counts.sum().item())
        if total_sends > 0:
            send_data_bytes = send_counts * cfg.hidden_dim * 2
            send_meta_bytes = send_counts * self.META_DIM * 4
            send_split_bytes = send_data_bytes + send_meta_bytes
            send_offsets = torch.cumsum(send_split_bytes, 0, dtype=torch.int32) - send_split_bytes
            send_data_offsets = send_offsets
            send_meta_offsets = send_offsets + send_data_bytes
            send_buf = torch.empty(send_split_bytes.sum().item(), dtype=torch.uint8, device=device)
            temp_send_counts = torch.zeros_like(send_counts)
            hip_module.hip_gather_and_pack(expert_y, expert_meta, expert_num_tokens, temp_send_counts, send_buf, send_data_offsets, send_meta_offsets, self.num_local_experts, self.max_recv, cfg.hidden_dim)
        else:
            send_buf = torch.empty(0, dtype=torch.uint8, device=device)
            send_split_bytes = torch.zeros_like(send_counts, dtype=torch.long)

        recv_data_bytes = recv_counts * cfg.hidden_dim * 2
        recv_meta_bytes = recv_counts * self.META_DIM * 4
        recv_split_bytes = recv_data_bytes + recv_meta_bytes
        recv_buf = torch.empty(recv_split_bytes.sum().item(), dtype=torch.uint8, device=device)
        dist.all_to_all_single(recv_buf, send_buf, recv_split_bytes.tolist(), send_split_bytes.tolist())

        if total_recv > 0:
            intermediate_buf = torch.empty(num_tokens, cfg.experts_per_token, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
            recv_offsets = torch.cumsum(recv_split_bytes, 0, dtype=torch.int32) - recv_split_bytes
            recv_data_byte_offsets = recv_offsets
            recv_meta_byte_offsets = recv_offsets + recv_data_bytes
            hip_module.hip_unpack_to_intermediate(recv_buf, intermediate_buf, recv_data_byte_offsets, recv_meta_byte_offsets, recv_counts, total_recv, cfg.hidden_dim, cfg.experts_per_token)
            hip_module.hip_final_reduction_smem(intermediate_buf, weights, out_tokens, num_tokens, cfg.hidden_dim, cfg.experts_per_token)

        return out_tokens

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    hip_all2all = HIPAllToAll(cfg, rank, world_size)
    expert_num, expert_x, expert_meta = hip_all2all.dispatch(rank_data.x, rank_data.indices)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
    y = torch.zeros(
        rank_data.num_tokens,
        cfg.hidden_dim,
        dtype=cfg.out_dtype,
        device=rank_data.x.device,
    )
    hip_all2all.combine(y, rank_data.weights, expert_meta, expert_y, expert_num, rank_data.num_tokens)
    return y

