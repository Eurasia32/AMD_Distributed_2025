import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# HIP kernel compilation setup
os.environ["CXX"] = "clang++"

CPP_WRAPPER = """
void hip_dispatch_tokens(torch::Tensor indices, torch::Tensor send_counts, torch::Tensor token_map,
                        torch::Tensor meta_data, int num_tokens, int experts_per_token,
                        int num_local_experts, int rank, int world_size);
void hip_combine_tokens(torch::Tensor recv_meta, torch::Tensor recv_buf, torch::Tensor weights,
                       torch::Tensor out_tokens, int total_recv, int meta_dim, int experts_per_token);
void hip_reorganize_expert_data(torch::Tensor recv_buf, torch::Tensor recv_meta,
                               torch::Tensor expert_x, torch::Tensor expert_meta,
                               torch::Tensor expert_num_tokens, int total_recv,
                               int num_local_experts, int hidden_dim, int max_recv, int meta_dim);
"""

CUDA_SRC = """
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

constexpr const int BLOCK_SIZE = 256;
constexpr const int WARP_SIZE = 64;

__global__ void dispatch_kernel(const int* indices, int* send_counts, int* token_map,
                               int* meta_data, int num_tokens, int experts_per_token,
                               int num_local_experts, int rank, int world_size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= num_tokens * experts_per_token) return;

    int token_idx = tid / experts_per_token;
    int expert_idx = tid % experts_per_token;

    int expert_id = indices[token_idx * experts_per_token + expert_idx];
    int dst_rank = expert_id / num_local_experts;

    // Atomic increment for send counts
    int old_count = atomicAdd(&send_counts[dst_rank], 1);

    // Store token mapping and metadata
    int meta_offset = tid * 5; // META_DIM = 5
    meta_data[meta_offset + 0] = expert_id;     // global expert id
    meta_data[meta_offset + 1] = rank;          // source rank
    meta_data[meta_offset + 2] = token_idx;     // source token index
    meta_data[meta_offset + 3] = expert_idx;    // expert index within token
    meta_data[meta_offset + 4] = 0;             // padding
}

__global__ void combine_kernel(const int* recv_meta, const __half* recv_buf,
                              const float* weights, __half* out_tokens,
                              int total_recv, int hidden_dim, int meta_dim, int experts_per_token) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= total_recv) return;

    // Extract metadata for this received token
    int src_token = recv_meta[tid * meta_dim + 2];
    int src_k = recv_meta[tid * meta_dim + 3];

    // Get weight for this expert - fix indexing to match reference
    float weight = weights[src_token * experts_per_token + src_k]; // weights shape: (num_tokens, experts_per_token)

    // Perform weighted accumulation with float32 intermediate
    for (int h = 0; h < hidden_dim; h++) {
        float value = __half2float(recv_buf[tid * hidden_dim + h]) * weight;
        float curr = __half2float(out_tokens[src_token * hidden_dim + h]);
        out_tokens[src_token * hidden_dim + h] = __float2half(curr + value);
    }
}

__global__ void reorganize_expert_kernel(const __half* recv_buf, const int* recv_meta,
                                        __half* expert_x, int* expert_meta,
                                        int* expert_num_tokens, int total_recv,
                                        int num_local_experts, int hidden_dim,
                                        int max_recv, int meta_dim) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= total_recv) return;

    // Get global expert ID and convert to local
    int global_eid = recv_meta[tid * meta_dim + 0];
    int local_eid = global_eid % num_local_experts;

    // Atomic increment to get position in expert buffer
    int pos = atomicAdd(&expert_num_tokens[local_eid], 1);

    if (pos < max_recv) {
        // Copy token data to expert buffer
        int expert_offset = local_eid * max_recv * hidden_dim + pos * hidden_dim;
        int recv_offset = tid * hidden_dim;

        for (int h = 0; h < hidden_dim; h++) {
            expert_x[expert_offset + h] = recv_buf[recv_offset + h];
        }

        // Copy metadata
        int meta_offset = local_eid * max_recv * meta_dim + pos * meta_dim;
        for (int m = 0; m < meta_dim; m++) {
            expert_meta[meta_offset + m] = recv_meta[tid * meta_dim + m];
        }
    }
}

// C++ wrapper functions
void hip_dispatch_tokens(torch::Tensor indices, torch::Tensor send_counts, torch::Tensor token_map,
                        torch::Tensor meta_data, int num_tokens, int experts_per_token,
                        int num_local_experts, int rank, int world_size) {
    int total_dispatches = num_tokens * experts_per_token;
    int blocks = (total_dispatches + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dispatch_kernel<<<blocks, BLOCK_SIZE>>>(
        indices.data_ptr<int>(),
        send_counts.data_ptr<int>(),
        token_map.data_ptr<int>(),
        meta_data.data_ptr<int>(),
        num_tokens, experts_per_token, num_local_experts, rank, world_size
    );
}

void hip_combine_tokens(torch::Tensor recv_meta, torch::Tensor recv_buf, torch::Tensor weights,
                       torch::Tensor out_tokens, int total_recv, int meta_dim, int experts_per_token) {
    int blocks = (total_recv + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int hidden_dim = recv_buf.size(1);

    combine_kernel<<<blocks, BLOCK_SIZE>>>(
        recv_meta.data_ptr<int>(),
        (__half*)recv_buf.data_ptr<at::Half>(),
        weights.data_ptr<float>(),
        (__half*)out_tokens.data_ptr<at::Half>(),
        total_recv, hidden_dim, meta_dim, experts_per_token
    );
}

void hip_reorganize_expert_data(torch::Tensor recv_buf, torch::Tensor recv_meta,
                               torch::Tensor expert_x, torch::Tensor expert_meta,
                               torch::Tensor expert_num_tokens, int total_recv,
                               int num_local_experts, int hidden_dim, int max_recv, int meta_dim) {
    int blocks = (total_recv + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reorganize_expert_kernel<<<blocks, BLOCK_SIZE>>>(
        (__half*)recv_buf.data_ptr<at::Half>(),
        recv_meta.data_ptr<int>(),
        (__half*)expert_x.data_ptr<at::Half>(),
        expert_meta.data_ptr<int>(),
        expert_num_tokens.data_ptr<int>(),
        total_recv, num_local_experts, hidden_dim, max_recv, meta_dim
    );
}
"""

# Compile HIP module
hip_module = load_inline(
    name='hip_all2all',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['hip_dispatch_tokens', 'hip_combine_tokens', 'hip_reorganize_expert_data'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)

class HIPAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # Convert indices to int32 for HIP kernel
        indices_int = indices.to(torch.int32)

        # Initialize buffers
        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        total_dispatches = indices.shape[0] * indices.shape[1]
        token_map = torch.zeros(total_dispatches, dtype=torch.int32, device=device)
        meta_data = torch.zeros(total_dispatches * self.META_DIM, dtype=torch.int32, device=device)

        # Launch HIP dispatch kernel
        hip_module.hip_dispatch_tokens(
            indices_int, send_counts, token_map, meta_data,
            indices.shape[0], indices.shape[1], self.num_local_experts,
            self.rank, self.world_size
        )

        # Convert send_counts to long for PyTorch all2all
        send_counts_long = send_counts.to(torch.long)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_long)

        # Prepare send buffers based on dispatch results
        # Create index mapping for gathering tokens
        token_indices = []
        meta_list = []
        current_idx = 0

        for rank in range(self.world_size):
            count = int(send_counts[rank].item())
            if count > 0:
                # Extract token indices for this rank
                rank_indices = []
                rank_meta = []

                for i in range(total_dispatches):
                    meta_offset = i * self.META_DIM
                    expert_id = int(meta_data[meta_offset].item())
                    dst_rank = expert_id // self.num_local_experts

                    if dst_rank == rank:
                        token_idx = int(meta_data[meta_offset + 2].item())
                        rank_indices.append(token_idx)
                        rank_meta.extend([
                            int(meta_data[meta_offset + j].item()) for j in range(self.META_DIM)
                        ])

                token_indices.extend(rank_indices[:count])
                meta_list.extend(rank_meta[:count * self.META_DIM])

        # Gather tokens using indices
        if token_indices:
            token_indices_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
            send_buf = dp_x[token_indices_tensor]
        else:
            send_buf = torch.empty((0, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)

        # Prepare metadata
        send_meta = torch.tensor(meta_list, dtype=torch.int32, device=device).view(-1, self.META_DIM)

        # All2All communication
        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        dist.all_to_all_single(
            recv_buf, send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_long.tolist(),
        )

        dist.all_to_all_single(
            recv_meta.view(-1), send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_long.tolist()],
        )

        # Reorganize into expert format using HIP kernel
        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim),
            dtype=cfg.in_dtype, device=device,
        )
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM),
            dtype=torch.int32, device=device,
        )

        if total_recv > 0:
            hip_module.hip_reorganize_expert_data(
                recv_buf, recv_meta, expert_x, expert_meta, expert_num_tokens,
                total_recv, self.num_local_experts, cfg.hidden_dim, self.max_recv, self.META_DIM
            )

        return expert_num_tokens, expert_x, expert_meta

    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor,
                expert_meta: torch.Tensor, expert_y: torch.Tensor,
                expert_num_tokens: torch.Tensor):
        device = out_tokens.device
        cfg = self.cfg

        # Prepare send buffers
        send_counts = [0] * self.world_size
        y_list = []
        meta_list = []

        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            for j in range(cnt):
                meta = expert_meta[local_eid, j]
                dst_rank = int(meta[1].item())
                send_counts[dst_rank] += 1
                y_list.append(expert_y[local_eid, j])
                meta_list.extend(meta.tolist())

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)

        # Prepare send data
        if y_list:
            send_buf = torch.stack(y_list, dim=0)
            send_meta = torch.tensor(meta_list, dtype=torch.int32, device=device).view(-1, self.META_DIM)
        else:
            send_buf = torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
            send_meta = torch.empty((0, self.META_DIM), dtype=torch.int32, device=device)

        # All2All communication
        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.META_DIM, dtype=torch.int32, device=device)

        dist.all_to_all_single(
            recv_buf, send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )

        dist.all_to_all_single(
            recv_meta.view(-1), send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )

        # Use HIP kernel for efficient combine operation
        if total_recv > 0:
            # Keep original types - kernel handles conversion internally
            weights_f32 = weights.to(torch.float32)

            hip_module.hip_combine_tokens(
                recv_meta, recv_buf, weights_f32, out_tokens,
                total_recv, self.META_DIM, cfg.experts_per_token
            )

        return out_tokens


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    hip_all2all = HIPAllToAll(cfg, rank, world_size)

    expert_num, expert_x, expert_meta = hip_all2all.dispatch(rank_data.x, rank_data.indices)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
    y = torch.zeros(
        cfg.max_num_tokens,
        cfg.hidden_dim,
        dtype=cfg.out_dtype,
        device=rank_data.x.device,
    )

    hip_all2all.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[: rank_data.num_tokens]