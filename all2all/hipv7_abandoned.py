import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# HIP kernel compilation setup
# 基于 hipv6 的代码进行 HIP Graph 改造。内核代码完全复用 hipv6 的版本。
os.environ["CXX"] = "clang++"

CPP_WRAPPER = """
void hip_count_and_map_dispatches(torch::Tensor indices, torch::Tensor send_counts, torch::Tensor dispatch_map,
                                  int num_tokens, int experts_per_token, int num_local_experts);
void hip_permute_and_pack(torch::Tensor original_tokens, torch::Tensor indices, torch::Tensor dispatch_map,
                           torch::Tensor temp_send_counts, torch::Tensor send_buf,
                           torch::Tensor send_data_offsets, torch::Tensor send_meta_offsets,
                           int num_tokens, int experts_per_token, int hidden_dim, int rank, int num_local_experts);
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
"""

# 使用与 hipv6 完全相同的内核源代码
CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

constexpr const int BLOCK_SIZE = 128;
constexpr const int META_DIM = 5;
constexpr const int WORLD_SIZE = 8;

#define VECTORIZE_COPY(dst, src, hidden_dim) \\
    do { \\
        const int4* src_vec = reinterpret_cast<const int4*>(src); \\
        int4* dst_vec = reinterpret_cast<int4*>(dst); \\
        for (int h = 0; h < (hidden_dim) / 8; ++h) { \\
            dst_vec[h] = src_vec[h]; \\
        } \\
    } while (0)

__global__ void count_and_map_kernel(const int* indices, int* send_counts, int* dispatch_map,
                                     int num_tokens, int experts_per_token, int num_local_experts) {
    __shared__ int s_send_counts[WORLD_SIZE];
    if (threadIdx.x < WORLD_SIZE) {
        s_send_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; 
         tid < num_tokens * experts_per_token; 
         tid += gridDim.x * blockDim.x) 
    {
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
    if (threadIdx.x < WORLD_SIZE) {
        atomicAdd(&send_counts[threadIdx.x], s_send_counts[threadIdx.x]);
    }
}


__global__ void permute_and_pack_kernel(const __half* original_tokens, const int* indices, const int* dispatch_map,
                                        int* temp_send_counts, unsigned char* send_buf,
                                        const int* send_data_offsets, const int* send_meta_offsets,
                                        int num_tokens, int experts_per_token, int hidden_dim, int rank, int num_local_experts) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_tokens * experts_per_token) return;

    int map_offset = tid * 3;
    int dst_rank = dispatch_map[map_offset + 0];
    int token_idx = dispatch_map[map_offset + 1];
    int expert_idx = dispatch_map[map_offset + 2];
    int pos = atomicAdd(&temp_send_counts[dst_rank], 1);

    __half* data_write_ptr = reinterpret_cast<__half*>(send_buf + send_data_offsets[dst_rank] + pos * hidden_dim * sizeof(__half));
    const __half* data_read_ptr = &original_tokens[token_idx * hidden_dim];
    VECTORIZE_COPY(data_write_ptr, data_read_ptr, hidden_dim);

    int* meta_write_ptr = reinterpret_cast<int*>(send_buf + send_meta_offsets[dst_rank] + pos * META_DIM * sizeof(int));
    int expert_id = indices[token_idx * experts_per_token + expert_idx];
    meta_write_ptr[0] = expert_id;
    meta_write_ptr[1] = rank;
    meta_write_ptr[2] = token_idx;
    meta_write_ptr[3] = expert_idx;
    meta_write_ptr[4] = 0;
}

__global__ void unpack_and_reorganize_kernel(const unsigned char* recv_buf, __half* expert_x, int* expert_meta,
                                             int* expert_num_tokens, const int* recv_data_byte_offsets,
                                             const int* recv_meta_byte_offsets, const int* recv_counts,
                                             int total_recv, int num_local_experts, int hidden_dim, int max_recv) {
    __shared__ int s_recv_offsets[WORLD_SIZE + 1];
    if (threadIdx.x == 0) {
        s_recv_offsets[0] = 0;
        for (int i = 0; i < WORLD_SIZE; ++i) {
            s_recv_offsets[i+1] = s_recv_offsets[i] + recv_counts[i];
        }
    }
    __syncthreads();

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_recv) return;

    int src_rank = 0;
    #pragma unroll
    for (int i = 1; i < WORLD_SIZE + 1; ++i) {
        if (tid >= s_recv_offsets[i]) {
            src_rank = i;
        }
    }

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
        for (int m = 0; m < META_DIM; ++m) {
            meta_write_ptr[m] = meta_read_ptr[m];
        }
    }
}

__global__ void count_backward_kernel(const int* expert_meta, const int* expert_num_tokens,
                                      int* send_counts, int num_local_experts, int max_recv) {
    int local_eid = blockIdx.x;
    int num_tokens_in_expert = expert_num_tokens[local_eid];
    int num_valid_tokens = num_tokens_in_expert < max_recv ? num_tokens_in_expert : max_recv;
    for (int i = threadIdx.x; i < num_valid_tokens; i += blockDim.x) {
        int dst_rank = expert_meta[local_eid * max_recv * META_DIM + i * META_DIM + 1];
        atomicAdd(&send_counts[dst_rank], 1);
    }
}

__global__ void gather_and_pack_kernel(const __half* expert_y, const int* expert_meta, const int* expert_num_tokens,
                                       int* temp_send_counts, unsigned char* send_buf,
                                       const int* send_data_offsets, const int* send_meta_offsets,
                                       int num_local_experts, int max_recv, int hidden_dim) {
    int local_eid = blockIdx.x;
    int token_idx_in_expert = threadIdx.x;
    int num_tokens_in_expert = expert_num_tokens[local_eid];
    int num_valid_tokens = num_tokens_in_expert < max_recv ? num_tokens_in_expert : max_recv;
    if (local_eid >= num_local_experts || token_idx_in_expert >= num_valid_tokens) return;
    const int* meta_read_ptr = &expert_meta[local_eid * max_recv * META_DIM + token_idx_in_expert * META_DIM];
    int dst_rank = meta_read_ptr[1];
    int pos = atomicAdd(&temp_send_counts[dst_rank], 1);
    const __half* data_read_ptr = &expert_y[local_eid * max_recv * hidden_dim + token_idx_in_expert * hidden_dim];
    __half* data_write_ptr = reinterpret_cast<__half*>(send_buf + send_data_offsets[dst_rank] + pos * hidden_dim * sizeof(__half));
    VECTORIZE_COPY(data_write_ptr, data_read_ptr, hidden_dim);
    int* meta_write_ptr = reinterpret_cast<int*>(send_buf + send_meta_offsets[dst_rank] + pos * META_DIM * sizeof(int));
    for (int m = 0; m < META_DIM; ++m) {
        meta_write_ptr[m] = meta_read_ptr[m];
    }
}

__global__ void unpack_to_intermediate_kernel(const unsigned char* recv_buf, __half* intermediate_buf,
                                              const int* recv_data_byte_offsets, const int* recv_meta_byte_offsets,
                                              const int* recv_counts, int total_recv, int hidden_dim, int experts_per_token) {
    __shared__ int s_recv_offsets[WORLD_SIZE + 1];
    if (threadIdx.x == 0) {
        s_recv_offsets[0] = 0;
        for (int i = 0; i < WORLD_SIZE; ++i) s_recv_offsets[i+1] = s_recv_offsets[i] + recv_counts[i];
    }
    __syncthreads();

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_recv) return;
    
    int src_rank = 0;
    #pragma unroll
    for (int i = 1; i < WORLD_SIZE + 1; ++i) if (tid >= s_recv_offsets[i]) src_rank = i;
    
    int intra_rank_idx = tid - s_recv_offsets[src_rank];
    const int* meta_read_ptr = reinterpret_cast<const int*>(recv_buf + recv_meta_byte_offsets[src_rank] + intra_rank_idx * META_DIM * sizeof(int));
    const __half* data_read_ptr = reinterpret_cast<const __half*>(recv_buf + recv_data_byte_offsets[src_rank] + intra_rank_idx * hidden_dim * sizeof(__half));

    int src_token = meta_read_ptr[2];
    int src_k = meta_read_ptr[3];

    __half* intermediate_write_ptr = &intermediate_buf[(src_token * experts_per_token + src_k) * hidden_dim];
    VECTORIZE_COPY(intermediate_write_ptr, data_read_ptr, hidden_dim);
}

__global__ void final_reduction_kernel_smem(const __half* intermediate_buf, const float* weights, __half* out_tokens,
                                            int num_tokens, int hidden_dim, int experts_per_token) {
    extern __shared__ float s_weights[];

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    if (threadIdx.x < experts_per_token) {
        s_weights[threadIdx.x] = weights[token_idx * experts_per_token + threadIdx.x];
    }
    __syncthreads();

    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int k = 0; k < experts_per_token; ++k) {
            float weight = s_weights[k];
            float val = __half2float(intermediate_buf[(token_idx * experts_per_token + k) * hidden_dim + h]);
            sum += val * weight;
        }
        out_tokens[token_idx * hidden_dim + h] = __float2half(sum);
    }
}
void hip_count_and_map_dispatches(torch::Tensor i, torch::Tensor sc, torch::Tensor dm, int n, int e, int l) { count_and_map_kernel<<<(n*e+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(i.data_ptr<int>(), sc.data_ptr<int>(), dm.data_ptr<int>(), n, e, l); }
void hip_permute_and_pack(torch::Tensor o, torch::Tensor i, torch::Tensor d, torch::Tensor t, torch::Tensor s, torch::Tensor sdo, torch::Tensor smo, int n, int e, int h, int r, int l) { permute_and_pack_kernel<<<(n*e+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>((const __half*)o.data_ptr<at::Half>(), i.data_ptr<int>(), d.data_ptr<int>(), t.data_ptr<int>(), (unsigned char*)s.data_ptr(), sdo.data_ptr<int>(), smo.data_ptr<int>(), n, e, h, r, l); }
void hip_unpack_and_reorganize(torch::Tensor r, torch::Tensor ex, torch::Tensor em, torch::Tensor en, torch::Tensor rdo, torch::Tensor rmo, torch::Tensor rc, int t, int l, int h, int mr) { unpack_and_reorganize_kernel<<<(t+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>((const unsigned char*)r.data_ptr(), (__half*)ex.data_ptr<at::Half>(), em.data_ptr<int>(), en.data_ptr<int>(), rdo.data_ptr<int>(), rmo.data_ptr<int>(), rc.data_ptr<int>(), t, l, h, mr); }
void hip_count_backward_sends(torch::Tensor em, torch::Tensor en, torch::Tensor sc, int l, int mr) { count_backward_kernel<<<l, BLOCK_SIZE>>>(em.data_ptr<int>(), en.data_ptr<int>(), sc.data_ptr<int>(), l, mr); }
void hip_gather_and_pack(torch::Tensor ey, torch::Tensor em, torch::Tensor en, torch::Tensor t, torch::Tensor s, torch::Tensor sdo, torch::Tensor smo, int l, int mr, int h) { gather_and_pack_kernel<<<l, BLOCK_SIZE>>>((const __half*)ey.data_ptr<at::Half>(), em.data_ptr<int>(), en.data_ptr<int>(), t.data_ptr<int>(), (unsigned char*)s.data_ptr(), sdo.data_ptr<int>(), smo.data_ptr<int>(), l, mr, h); }
void hip_unpack_to_intermediate(torch::Tensor rb, torch::Tensor ib, torch::Tensor rdo, torch::Tensor rmo, torch::Tensor rc, int t, int h, int e) { unpack_to_intermediate_kernel<<<(t+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>((const unsigned char*)rb.data_ptr(), (__half*)ib.data_ptr<at::Half>(), rdo.data_ptr<int>(), rmo.data_ptr<int>(), rc.data_ptr<int>(), t, h, e); }
void hip_final_reduction_smem(torch::Tensor ib, torch::Tensor w, torch::Tensor ot, int n, int h, int e) { size_t smem = e * sizeof(float); final_reduction_kernel_smem<<<n, BLOCK_SIZE, smem>>>((const __half*)ib.data_ptr<at::Half>(), w.data_ptr<float>(), (__half*)ot.data_ptr<at::Half>(), n, h, e); }
"""

# Compile HIP module
hip_module = load_inline(
    name='hip_all2all_fused_v20_graph_from_v6', # Renamed to avoid cache conflicts
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=[
        'hip_count_and_map_dispatches', 'hip_permute_and_pack', 'hip_unpack_and_reorganize',
        'hip_count_backward_sends', 'hip_gather_and_pack', 
        'hip_unpack_to_intermediate', 'hip_final_reduction_smem'
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
        
        self.is_captured = False
        self.graph_pre_comm_dispatch = None
        self.graph_pre_comm_combine = None
        
        self.static_tensors = {}

    def _setup_static_tensors(self, device):
        cfg = self.cfg
        max_tokens = cfg.max_num_tokens
        
        self.static_tensors['dp_x'] = torch.empty(max_tokens, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        self.static_tensors['indices'] = torch.empty(max_tokens, cfg.experts_per_token, dtype=torch.int32, device=device)
        self.static_tensors['weights'] = torch.empty(max_tokens, cfg.experts_per_token, dtype=torch.float32, device=device)
        self.static_tensors['out_tokens'] = torch.empty(max_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        
        self.static_tensors['send_counts_dispatch'] = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        self.static_tensors['dispatch_map'] = torch.empty(max_tokens * cfg.experts_per_token, 3, dtype=torch.int32, device=device)
        
        max_buffer_items = max_tokens * cfg.experts_per_token 
        buffer_size = max_buffer_items * (cfg.hidden_dim * 2 + self.META_DIM * 4) 
        self.static_tensors['send_buf_dispatch'] = torch.empty(buffer_size, dtype=torch.uint8, device=device)
        self.static_tensors['recv_buf_dispatch'] = torch.empty(buffer_size, dtype=torch.uint8, device=device)
        self.static_tensors['temp_send_counts_dispatch'] = torch.zeros(self.world_size, dtype=torch.int32, device=device)

        self.static_tensors['expert_num_tokens'] = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        self.static_tensors['expert_x'] = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        self.static_tensors['expert_meta'] = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM), dtype=torch.int32, device=device)
        self.static_tensors['expert_y'] = torch.empty_like(self.static_tensors['expert_x'], dtype=cfg.out_dtype)

        self.static_tensors['send_counts_combine'] = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        self.static_tensors['send_buf_combine'] = torch.empty(buffer_size, dtype=torch.uint8, device=device)
        self.static_tensors['recv_buf_combine'] = torch.empty(buffer_size, dtype=torch.uint8, device=device)
        self.static_tensors['temp_send_counts_combine'] = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        self.static_tensors['intermediate_buf'] = torch.empty(max_tokens, cfg.experts_per_token, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)

    def capture_graphs(self, device):
        self._setup_static_tensors(device)
        st = self.static_tensors
        cfg = self.cfg
        
        torch.cuda.synchronize(device)

        # --- Warmup runs on the default stream ---
        st['send_counts_dispatch'].zero_()
        hip_module.hip_count_and_map_dispatches(st['indices'], st['send_counts_dispatch'], st['dispatch_map'], cfg.max_num_tokens, cfg.experts_per_token, self.num_local_experts)
        st['send_counts_combine'].zero_()
        hip_module.hip_count_backward_sends(st['expert_meta'], st['expert_num_tokens'], st['send_counts_combine'], self.num_local_experts, self.max_recv)
        torch.cuda.synchronize(device)

        # === Capture Dispatch Pre-Communication Graph on the default stream ===
        self.graph_pre_comm_dispatch = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph_pre_comm_dispatch):
            st['send_counts_dispatch'].zero_()
            hip_module.hip_count_and_map_dispatches(st['indices'], st['send_counts_dispatch'], st['dispatch_map'], cfg.max_num_tokens, cfg.experts_per_token, self.num_local_experts)

        # === Capture Combine Pre-Communication Graph on the default stream ===
        self.graph_pre_comm_combine = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph_pre_comm_combine):
            st['send_counts_combine'].zero_()
            hip_module.hip_count_backward_sends(st['expert_meta'], st['expert_num_tokens'], st['send_counts_combine'], self.num_local_experts, self.max_recv)
        
        torch.cuda.synchronize()
        self.is_captured = True


# --- Main kernel execution function ---
hip_a2a_cache = {}

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    cache_key = (rank, world_size, cfg.num_experts, cfg.experts_per_token, cfg.hidden_dim, cfg.max_num_tokens)
    if cache_key not in hip_a2a_cache:
        hip_a2a_cache[cache_key] = HIPAllToAll(cfg, rank, world_size)
    hip_all2all = hip_a2a_cache[cache_key]

    if not hip_all2all.is_captured:
        hip_all2all.capture_graphs(device)

    st = hip_all2all.static_tensors
    num_tokens = rank_data.num_tokens

    # === 1. Copy dynamic input data to static tensors ===
    st['dp_x'].zero_()
    st['dp_x'][:num_tokens].copy_(rank_data.x)
    st['indices'].zero_()
    st['indices'][:num_tokens].copy_(rank_data.indices.to(torch.int32))
    
    # === DISPATCH PHASE ===
    # 2. Replay pre-communication graph (count & map)
    hip_all2all.graph_pre_comm_dispatch.replay()
    
    # 3. CPU-side logic and communication for counts
    send_counts = st['send_counts_dispatch']
    recv_counts_t = torch.empty_like(send_counts, dtype=torch.long)
    dist.all_to_all_single(recv_counts_t, send_counts.to(torch.long))
    recv_counts = recv_counts_t.to(torch.int32)
    
    # 4. Pack kernel (dynamic offsets, cannot be in graph)
    send_data_bytes = send_counts * cfg.hidden_dim * 2 
    send_meta_bytes = hip_all2all.META_DIM * 4 * send_counts
    send_split_bytes = send_data_bytes + send_meta_bytes
    send_data_offsets = torch.cumsum(send_split_bytes, 0, dtype=torch.int32) - send_split_bytes
    send_meta_offsets = send_data_offsets + send_data_bytes
    
    st['temp_send_counts_dispatch'].zero_()
    hip_module.hip_permute_and_pack(st['dp_x'], st['indices'], st['dispatch_map'], st['temp_send_counts_dispatch'], st['send_buf_dispatch'], send_data_offsets, send_meta_offsets, num_tokens, cfg.experts_per_token, cfg.hidden_dim, rank, hip_all2all.num_local_experts)

    # 5. All-to-all communication for data
    recv_data_bytes = recv_counts * cfg.hidden_dim * 2
    recv_meta_bytes = hip_all2all.META_DIM * 4 * recv_counts
    recv_split_bytes = recv_data_bytes + recv_meta_bytes
    
    total_send_bytes = int(send_split_bytes.sum().item())
    total_recv_bytes = int(recv_split_bytes.sum().item())
    
    send_buf_view = st['send_buf_dispatch'][:total_send_bytes]
    recv_buf_view = st['recv_buf_dispatch'][:total_recv_bytes]
    dist.all_to_all_single(recv_buf_view, send_buf_view, recv_split_bytes.tolist(), send_split_bytes.tolist())
    
    # 6. Unpack kernel (dynamic total_recv, cannot be in graph)
    st['expert_num_tokens'].zero_()
    total_recv = int(recv_counts.sum().item())
    if total_recv > 0:
        recv_data_byte_offsets = torch.cumsum(recv_split_bytes, 0, dtype=torch.int32) - recv_split_bytes
        recv_meta_byte_offsets = recv_data_byte_offsets + recv_data_bytes
        hip_module.hip_unpack_and_reorganize(recv_buf_view, st['expert_x'], st['expert_meta'], st['expert_num_tokens'], recv_data_byte_offsets, recv_meta_byte_offsets, recv_counts, total_recv, hip_all2all.num_local_experts, cfg.hidden_dim, hip_all2all.max_recv)
    
    # === EXPERT COMPUTATION ===
    st['expert_y'].copy_(st['expert_x'].to(cfg.out_dtype) * (1 + rank))

    # === COMBINE PHASE ===
    st['weights'].zero_()
    st['weights'][:num_tokens].copy_(rank_data.weights)
    
    # 7. Replay pre-comm graph for combine
    hip_all2all.graph_pre_comm_combine.replay()

    # 8. CPU-side logic and comm for combine counts
    send_counts_combine = st['send_counts_combine']
    recv_counts_combine_t = torch.empty_like(send_counts_combine, dtype=torch.long)
    dist.all_to_all_single(recv_counts_combine_t, send_counts_combine.to(torch.long))
    recv_counts_combine = recv_counts_combine_t.to(torch.int32)

    # 9. Pack kernel for combine
    send_data_bytes = send_counts_combine * cfg.hidden_dim * 2
    send_meta_bytes = hip_all2all.META_DIM * 4 * send_counts_combine
    send_split_bytes = send_data_bytes + send_meta_bytes
    send_data_offsets = torch.cumsum(send_split_bytes, 0, dtype=torch.int32) - send_split_bytes
    send_meta_offsets = send_data_offsets + send_data_bytes
    
    st['temp_send_counts_combine'].zero_()
    hip_module.hip_gather_and_pack(st['expert_y'], st['expert_meta'], st['expert_num_tokens'], st['temp_send_counts_combine'], st['send_buf_combine'], send_data_offsets, send_meta_offsets, hip_all2all.num_local_experts, hip_all2all.max_recv, cfg.hidden_dim)
    
    # 10. Comm for combine data
    recv_data_bytes = recv_counts_combine * cfg.hidden_dim * 2
    recv_meta_bytes = hip_all2all.META_DIM * 4 * recv_counts_combine
    recv_split_bytes = recv_data_bytes + recv_meta_bytes
    
    total_send_bytes = int(send_split_bytes.sum().item())
    total_recv_bytes = int(recv_split_bytes.sum().item())

    send_buf_view = st['send_buf_combine'][:total_send_bytes]
    recv_buf_view = st['recv_buf_combine'][:total_recv_bytes]
    dist.all_to_all_single(recv_buf_view, send_buf_view, recv_split_bytes.tolist(), send_split_bytes.tolist())
    
    # 11. Final unpack and reduction kernels
    st['out_tokens'].zero_()
    total_recv_combine = int(recv_counts_combine.sum().item())
    if total_recv_combine > 0:
        recv_data_byte_offsets = torch.cumsum(recv_split_bytes, 0, dtype=torch.int32) - recv_split_bytes
        recv_meta_byte_offsets = recv_data_byte_offsets + recv_data_bytes
        hip_module.hip_unpack_to_intermediate(recv_buf_view, st['intermediate_buf'], recv_data_byte_offsets, recv_meta_byte_offsets, recv_counts_combine, total_recv_combine, cfg.hidden_dim, cfg.experts_per_token)
        hip_module.hip_final_reduction_smem(st['intermediate_buf'], st['weights'], st['out_tokens'], num_tokens, cfg.hidden_dim, cfg.experts_per_token)

    return st['out_tokens'][:num_tokens].clone()

