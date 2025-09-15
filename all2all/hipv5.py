import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# HIP kernel compilation setup
os.environ["CXX"] = "clang++"

CPP_WRAPPER = """
// Forward declarations for the final kernel fusion implementation
void hip_count_and_map_dispatches(torch::Tensor indices, torch::Tensor send_counts, torch::Tensor dispatch_map,
                                  int num_tokens, int experts_per_token, int num_local_experts);

void hip_permute_tokens(torch::Tensor original_tokens, torch::Tensor indices, torch::Tensor dispatch_map,
                        torch::Tensor send_offsets, torch::Tensor temp_send_counts,
                        torch::Tensor send_buf, torch::Tensor send_meta,
                        int num_tokens, int experts_per_token, int hidden_dim,
                        int rank, int num_local_experts);

void hip_reorganize_expert_data(torch::Tensor recv_buf, torch::Tensor recv_meta,
                               torch::Tensor expert_x, torch::Tensor expert_meta,
                               torch::Tensor expert_num_tokens, int total_recv,
                               int num_local_experts, int hidden_dim, int max_recv, int meta_dim);

void hip_count_backward_sends(torch::Tensor expert_meta, torch::Tensor expert_num_tokens,
                              torch::Tensor send_counts, int num_local_experts, int max_recv);

void hip_create_gather_maps(torch::Tensor expert_num_tokens, torch::Tensor expert_offsets,
                            torch::Tensor token_to_expert_map, torch::Tensor token_to_idx_in_expert_map,
                            int num_local_experts);

void hip_gather_flat(torch::Tensor expert_y, torch::Tensor expert_meta,
                     torch::Tensor token_to_expert_map, torch::Tensor token_to_idx_in_expert_map,
                     torch::Tensor send_offsets, torch::Tensor temp_send_counts,
                     torch::Tensor send_buf, torch::Tensor send_meta,
                     int total_sends, int max_recv, int hidden_dim, int meta_dim);

void hip_combine_tokens(torch::Tensor recv_meta, torch::Tensor recv_buf, torch::Tensor weights,
                       torch::Tensor out_tokens, int total_recv, int meta_dim, int experts_per_token);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

constexpr const int BLOCK_SIZE = 256;
constexpr const int META_DIM = 5;

// --- Helper for vectorized copy ---
#define VECTORIZE_COPY(dst, src, hidden_dim) \
    do { \
        const int4* src_vec = reinterpret_cast<const int4*>(src); \
        int4* dst_vec = reinterpret_cast<int4*>(dst); \
        for (int h = 0; h < (hidden_dim) / 8; ++h) { \
            dst_vec[h] = src_vec[h]; \
        } \
    } while (0)

// --- Dispatch Kernels (Unchanged) ---
__global__ void count_and_map_kernel(const int* indices, int* send_counts, int* dispatch_map,
                                     int num_tokens, int experts_per_token, int num_local_experts) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_tokens * experts_per_token) return;
    int token_idx = tid / experts_per_token;
    int expert_idx = tid % experts_per_token;
    int expert_id = indices[token_idx * experts_per_token + expert_idx];
    int dst_rank = expert_id / num_local_experts;
    atomicAdd(&send_counts[dst_rank], 1);
    int map_offset = tid * 3;
    dispatch_map[map_offset + 0] = dst_rank;
    dispatch_map[map_offset + 1] = token_idx;
    dispatch_map[map_offset + 2] = expert_idx;
}

__global__ void permute_kernel(const __half* original_tokens, const int* indices, const int* dispatch_map,
                               const int* send_offsets, int* temp_send_counts,
                               __half* send_buf, int* send_meta,
                               int num_tokens, int experts_per_token, int hidden_dim,
                               int rank, int num_local_experts) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_tokens * experts_per_token) return;
    int map_offset = tid * 3;
    int dst_rank = dispatch_map[map_offset + 0];
    int token_idx = dispatch_map[map_offset + 1];
    int expert_idx = dispatch_map[map_offset + 2];
    int pos = atomicAdd(&temp_send_counts[dst_rank], 1);
    int final_pos = send_offsets[dst_rank] + pos;
    VECTORIZE_COPY(&send_buf[final_pos * hidden_dim], &original_tokens[token_idx * hidden_dim], hidden_dim);
    int meta_offset = final_pos * META_DIM;
    int expert_id = indices[token_idx * experts_per_token + expert_idx];
    send_meta[meta_offset + 0] = expert_id;
    send_meta[meta_offset + 1] = rank;
    send_meta[meta_offset + 2] = token_idx;
    send_meta[meta_offset + 3] = expert_idx;
    send_meta[meta_offset + 4] = 0;
}

__global__ void reorganize_expert_kernel(const __half* recv_buf, const int* recv_meta,
                                        __half* expert_x, int* expert_meta,
                                        int* expert_num_tokens, int total_recv,
                                        int num_local_experts, int hidden_dim,
                                        int max_recv, int meta_dim) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_recv) return;
    int global_eid = recv_meta[tid * meta_dim + 0];
    int local_eid = global_eid % num_local_experts;
    int pos = atomicAdd(&expert_num_tokens[local_eid], 1);
    if (pos < max_recv) {
        VECTORIZE_COPY(&expert_x[local_eid * max_recv * hidden_dim + pos * hidden_dim], &recv_buf[tid * hidden_dim], hidden_dim);
        for (int m = 0; m < meta_dim; m++) {
            expert_meta[local_eid * max_recv * meta_dim + pos * meta_dim + m] = recv_meta[tid * meta_dim + m];
        }
    }
}


// --- Combine Kernels (Optimized Scheduling) ---

__global__ void count_backward_kernel(const int* expert_meta, const int* expert_num_tokens,
                                      int* send_counts, int num_local_experts, int max_recv) {
    int local_eid = threadIdx.x + blockDim.x * blockIdx.x;
    if (local_eid >= num_local_experts) return;
    int num_valid_tokens = expert_num_tokens[local_eid];
    for (int i = 0; i < num_valid_tokens; ++i) {
        int dst_rank = expert_meta[local_eid * max_recv * META_DIM + i * META_DIM + 1];
        atomicAdd(&send_counts[dst_rank], 1);
    }
}

// New Kernel to replace PyTorch-based map creation
__global__ void create_gather_maps_kernel(const int* expert_num_tokens, const int* expert_offsets,
                                          int* token_to_expert_map, int* token_to_idx_in_expert_map,
                                          int num_local_experts) {
    int local_eid = threadIdx.x + blockDim.x * blockIdx.x;
    if (local_eid >= num_local_experts) return;

    int num_tokens_in_expert = expert_num_tokens[local_eid];
    int global_offset = expert_offsets[local_eid];

    for (int i = 0; i < num_tokens_in_expert; ++i) {
        token_to_expert_map[global_offset + i] = local_eid;
        token_to_idx_in_expert_map[global_offset + i] = i;
    }
}

__global__ void gather_flat_kernel(const __half* expert_y, const int* expert_meta,
                                   const int* token_to_expert_map, const int* token_to_idx_in_expert_map,
                                   const int* send_offsets, int* temp_send_counts,
                                   __half* send_buf, int* send_meta,
                                   int total_sends, int max_recv, int hidden_dim, int meta_dim) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_sends) return;

    int local_eid = token_to_expert_map[tid];
    int token_idx_in_expert = token_to_idx_in_expert_map[tid];

    const int* meta_read_ptr = &expert_meta[local_eid * max_recv * meta_dim + token_idx_in_expert * meta_dim];
    int dst_rank = meta_read_ptr[1];
    int pos = atomicAdd(&temp_send_counts[dst_rank], 1);
    int final_pos = send_offsets[dst_rank] + pos;

    VECTORIZE_COPY(&send_buf[final_pos * hidden_dim], &expert_y[local_eid * max_recv * hidden_dim + token_idx_in_expert * hidden_dim], hidden_dim);
    
    int* meta_write_ptr = &send_meta[final_pos * meta_dim];
    for (int m = 0; m < meta_dim; ++m) {
        meta_write_ptr[m] = meta_read_ptr[m];
    }
}

__global__ void combine_kernel(const int* recv_meta, const __half* recv_buf,
                              const float* weights, float* out_tokens,
                              int total_recv, int hidden_dim, int meta_dim, int experts_per_token) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= total_recv) return;
    int src_token = recv_meta[tid * meta_dim + 2];
    int src_k = recv_meta[tid * meta_dim + 3];
    float weight = weights[src_token * experts_per_token + src_k];
    for (int h = 0; h < hidden_dim; h++) {
        float val = __half2float(recv_buf[tid * hidden_dim + h]) * weight;
        atomicAdd(&out_tokens[src_token * hidden_dim + h], val);
    }
}

// --- C++ Wrapper Functions ---
void hip_count_and_map_dispatches(torch::Tensor indices, torch::Tensor send_counts, torch::Tensor dispatch_map, int num_tokens, int experts_per_token, int num_local_experts) {
    int blocks = (num_tokens * experts_per_token + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_and_map_kernel<<<blocks, BLOCK_SIZE>>>(indices.data_ptr<int>(), send_counts.data_ptr<int>(), dispatch_map.data_ptr<int>(), num_tokens, experts_per_token, num_local_experts);
}
void hip_permute_tokens(torch::Tensor original_tokens, torch::Tensor indices, torch::Tensor dispatch_map, torch::Tensor send_offsets, torch::Tensor temp_send_counts, torch::Tensor send_buf, torch::Tensor send_meta, int num_tokens, int experts_per_token, int hidden_dim, int rank, int num_local_experts) {
    int blocks = (num_tokens * experts_per_token + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute_kernel<<<blocks, BLOCK_SIZE>>>((const __half*)original_tokens.data_ptr<at::Half>(), indices.data_ptr<int>(), dispatch_map.data_ptr<int>(), send_offsets.data_ptr<int>(), temp_send_counts.data_ptr<int>(), (__half*)send_buf.data_ptr<at::Half>(), send_meta.data_ptr<int>(), num_tokens, experts_per_token, hidden_dim, rank, num_local_experts);
}
void hip_reorganize_expert_data(torch::Tensor recv_buf, torch::Tensor recv_meta, torch::Tensor expert_x, torch::Tensor expert_meta, torch::Tensor expert_num_tokens, int total_recv, int num_local_experts, int hidden_dim, int max_recv, int meta_dim) {
    int blocks = (total_recv + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reorganize_expert_kernel<<<blocks, BLOCK_SIZE>>>((const __half*)recv_buf.data_ptr<at::Half>(), recv_meta.data_ptr<int>(), (__half*)expert_x.data_ptr<at::Half>(), expert_meta.data_ptr<int>(), expert_num_tokens.data_ptr<int>(), total_recv, num_local_experts, hidden_dim, max_recv, meta_dim);
}
void hip_count_backward_sends(torch::Tensor expert_meta, torch::Tensor expert_num_tokens, torch::Tensor send_counts, int num_local_experts, int max_recv) {
    int blocks = (num_local_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_backward_kernel<<<blocks, BLOCK_SIZE>>>(expert_meta.data_ptr<int>(), expert_num_tokens.data_ptr<int>(), send_counts.data_ptr<int>(), num_local_experts, max_recv);
}
void hip_create_gather_maps(torch::Tensor expert_num_tokens, torch::Tensor expert_offsets, torch::Tensor token_to_expert_map, torch::Tensor token_to_idx_in_expert_map, int num_local_experts) {
    int blocks = (num_local_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    create_gather_maps_kernel<<<blocks, BLOCK_SIZE>>>(expert_num_tokens.data_ptr<int>(), expert_offsets.data_ptr<int>(), token_to_expert_map.data_ptr<int>(), token_to_idx_in_expert_map.data_ptr<int>(), num_local_experts);
}
void hip_gather_flat(torch::Tensor expert_y, torch::Tensor expert_meta, torch::Tensor token_to_expert_map, torch::Tensor token_to_idx_in_expert_map, torch::Tensor send_offsets, torch::Tensor temp_send_counts, torch::Tensor send_buf, torch::Tensor send_meta, int total_sends, int max_recv, int hidden_dim, int meta_dim) {
    int blocks = (total_sends + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gather_flat_kernel<<<blocks, BLOCK_SIZE>>>((const __half*)expert_y.data_ptr<at::Half>(), expert_meta.data_ptr<int>(), token_to_expert_map.data_ptr<int>(), token_to_idx_in_expert_map.data_ptr<int>(), send_offsets.data_ptr<int>(), temp_send_counts.data_ptr<int>(), (__half*)send_buf.data_ptr<at::Half>(), send_meta.data_ptr<int>(), total_sends, max_recv, hidden_dim, meta_dim);
}
void hip_combine_tokens(torch::Tensor recv_meta, torch::Tensor recv_buf, torch::Tensor weights, torch::Tensor out_tokens, int total_recv, int meta_dim, int experts_per_token) {
    int blocks = (total_recv + BLOCK_SIZE - 1) / BLOCK_SIZE;
    combine_kernel<<<blocks, BLOCK_SIZE>>>(recv_meta.data_ptr<int>(), (const __half*)recv_buf.data_ptr<at::Half>(), weights.data_ptr<float>(), out_tokens.data_ptr<float>(), total_recv, recv_buf.size(1), meta_dim, experts_per_token);
}
"""

# Compile HIP module
hip_module = load_inline(
    name='hip_all2all_kernel_fusion',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=[
        'hip_count_and_map_dispatches', 'hip_permute_tokens', 'hip_reorganize_expert_data',
        'hip_count_backward_sends', 'hip_create_gather_maps', 'hip_gather_flat', 'hip_combine_tokens'
    ],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
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

        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        dispatch_map = torch.empty(total_dispatches, 3, dtype=torch.int32, device=device)
        hip_module.hip_count_and_map_dispatches(indices_int, send_counts, dispatch_map, num_tokens, experts_per_token, self.num_local_experts)

        send_counts_long = send_counts.to(torch.long)
        recv_counts_t = torch.empty_like(send_counts_long)
        dist.all_to_all_single(recv_counts_t, send_counts_long)
        total_recv = int(recv_counts_t.sum().item())

        send_offsets = torch.cumsum(send_counts, dim=0, dtype=torch.int32) - send_counts
        temp_send_counts = torch.zeros_like(send_counts)
        send_buf = torch.empty(total_dispatches, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        send_meta = torch.empty(total_dispatches, self.META_DIM, dtype=torch.int32, device=device)
        hip_module.hip_permute_tokens(dp_x, indices_int, dispatch_map, send_offsets, temp_send_counts, send_buf, send_meta, num_tokens, experts_per_token, cfg.hidden_dim, self.rank, self.num_local_experts)

        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)
        
        dist.all_to_all_single(recv_buf, send_buf, recv_counts_t.tolist(), send_counts_long.tolist())
        dist.all_to_all_single(recv_meta.view(-1), send_meta.view(-1), [c * self.META_DIM for c in recv_counts_t.tolist()], [c * self.META_DIM for c in send_counts_long.tolist()])

        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM), dtype=torch.int32, device=device)
        if total_recv > 0:
            hip_module.hip_reorganize_expert_data(recv_buf, recv_meta, expert_x, expert_meta, expert_num_tokens, total_recv, self.num_local_experts, cfg.hidden_dim, self.max_recv, self.META_DIM)

        return expert_num_tokens, expert_x, expert_meta

    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, expert_meta: torch.Tensor, expert_y: torch.Tensor, expert_num_tokens: torch.Tensor):
        device = out_tokens.device
        cfg = self.cfg

        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        hip_module.hip_count_backward_sends(expert_meta, expert_num_tokens, send_counts, self.num_local_experts, self.max_recv)

        send_counts_long = send_counts.to(torch.long)
        recv_counts_t = torch.empty_like(send_counts_long)
        dist.all_to_all_single(recv_counts_t, send_counts_long)
        total_recv = int(recv_counts_t.sum().item())

        total_sends = int(expert_num_tokens.sum().item())
        if total_sends > 0:
            send_offsets = torch.cumsum(send_counts, dim=0, dtype=torch.int32) - send_counts
            temp_send_counts = torch.zeros_like(send_counts)
            
            # Use a dedicated kernel to create the gather maps efficiently.
            expert_offsets = (torch.cumsum(expert_num_tokens, 0) - expert_num_tokens).to(torch.int32)
            token_to_expert_map = torch.empty(total_sends, device=device, dtype=torch.int32)
            token_to_idx_in_expert_map = torch.empty(total_sends, device=device, dtype=torch.int32)
            hip_module.hip_create_gather_maps(expert_num_tokens, expert_offsets, token_to_expert_map, token_to_idx_in_expert_map, self.num_local_experts)

            send_buf = torch.empty(total_sends, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
            send_meta = torch.empty(total_sends, self.META_DIM, dtype=torch.int32, device=device)
            
            hip_module.hip_gather_flat(expert_y, expert_meta, token_to_expert_map, token_to_idx_in_expert_map, send_offsets, temp_send_counts, send_buf, send_meta, total_sends, self.max_recv, cfg.hidden_dim, self.META_DIM)
        else: 
             send_buf = torch.empty(0, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
             send_meta = torch.empty(0, self.META_DIM, dtype=torch.int32, device=device)


        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)
        
        dist.all_to_all_single(recv_buf, send_buf, recv_counts_t.tolist(), send_counts_long.tolist())
        dist.all_to_all_single(recv_meta.view(-1), send_meta.view(-1), [c * self.META_DIM for c in recv_counts_t.tolist()], [c * self.META_DIM for c in send_counts_long.tolist()])

        if total_recv > 0:
            weights_f32 = weights.to(torch.float32)
            out_tokens_f32 = torch.zeros_like(out_tokens, dtype=torch.float32)
            hip_module.hip_combine_tokens(recv_meta, recv_buf, weights_f32, out_tokens_f32, total_recv, self.META_DIM, cfg.experts_per_token)
            out_tokens.copy_(out_tokens_f32.to(out_tokens.dtype))

        return out_tokens

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    hip_all2all = HIPAllToAll(cfg, rank, world_size)
    expert_num, expert_x, expert_meta = hip_all2all.dispatch(rank_data.x, rank_data.indices)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
    y = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=cfg.out_dtype, device=rank_data.x.device)
    hip_all2all.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)
    return y[: rank_data.num_tokens]

