import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# HIP kernel编译设置
os.environ["CXX"] = "clang++"

# 核心改动：
# 1. (本次修改) 双重加固内核安全性：在 `unpack_and_reorganize` 和 `fused_unpack_and_combine` 两个
#    解析recv_buf的关键内核中，都加入了对 `intra_rank_idx` 的严格边界检查，
#    彻底杜绝因索引计算错误导致的 "Memory access fault"。
# 2. 修复异步执行中的悬垂指针问题。
# 3. 引入HIP Stream (torch.cuda.Stream) 实现异步执行。
# 4. 回归到hipv7的单内核融合Combine方案。
# 5. 在Python宿主代码中精心设计同步点。
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
void hip_fused_unpack_and_combine(torch::Tensor recv_buf, torch::Tensor weights, torch::Tensor out_tokens_f32,
                                  torch::Tensor recv_data_byte_offsets, torch::Tensor recv_meta_byte_offsets,
                                  torch::Tensor recv_counts, int total_recv, int hidden_dim, int experts_per_token, int num_tokens);
"""

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
        _Pragma("unroll") \\
        for (int h = 0; h < (hidden_dim) / 8; ++h) { \\
            dst_vec[h] = src_vec[h]; \\
        } \\
    } while (0)

__global__ void count_and_map_kernel(const int* indices, int* send_counts, int* dispatch_map,
                                     int num_tokens, int experts_per_token, int num_local_experts) {
    __shared__ int s_send_counts[WORLD_SIZE];
    if (threadIdx.x < WORLD_SIZE) s_send_counts[threadIdx.x] = 0;
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
    if (threadIdx.x < WORLD_SIZE) atomicAdd(&send_counts[threadIdx.x], s_send_counts[threadIdx.x]);
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
    meta_write_ptr[0] = expert_id; meta_write_ptr[1] = rank; meta_write_ptr[2] = token_idx; meta_write_ptr[3] = expert_idx; meta_write_ptr[4] = 0;
}

// *** 增加安全检查 ***
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
    #pragma unroll
    for (int i = 1; i < WORLD_SIZE + 1; ++i) if (tid >= s_recv_offsets[i]) src_rank = i;
    
    int intra_rank_idx = tid - s_recv_offsets[src_rank];
    // *** 新增边界检查 ***
    if(intra_rank_idx < 0 || intra_rank_idx >= recv_counts[src_rank]) return;

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
    for (int m = 0; m < META_DIM; ++m) meta_write_ptr[m] = meta_read_ptr[m];
}

// *** 增加安全检查 ***
__global__ void fused_unpack_and_combine_kernel(
    const unsigned char* recv_buf,
    const float* weights,
    float* out_tokens_f32,
    const int* recv_data_byte_offsets,
    const int* recv_meta_byte_offsets,
    const int* recv_counts,
    int total_recv,
    int hidden_dim,
    int experts_per_token,
    int num_tokens)
{
    __shared__ int s_recv_offsets[WORLD_SIZE + 1];
    if (threadIdx.x == 0) {
        s_recv_offsets[0] = 0;
        for (int i = 0; i < WORLD_SIZE; ++i) s_recv_offsets[i+1] = s_recv_offsets[i] + recv_counts[i];
    }
    __syncthreads();

    int tid = blockIdx.x;
    if (tid >= total_recv) return;

    __shared__ int s_src_token;
    __shared__ float s_weight;
    __shared__ bool s_is_valid;

    if (threadIdx.x == 0) {
        s_is_valid = false;
        int src_rank = 0;
        #pragma unroll
        for (int i = 1; i < WORLD_SIZE + 1; ++i) if (tid >= s_recv_offsets[i]) src_rank = i;
        
        int intra_rank_idx = tid - s_recv_offsets[src_rank];
        // *** 新增边界检查 ***
        if(intra_rank_idx < 0 || intra_rank_idx >= recv_counts[src_rank]) {
             // Do nothing if index is out of bounds
        } else {
            const int* meta_read_ptr = reinterpret_cast<const int*>(recv_buf + recv_meta_byte_offsets[src_rank] + intra_rank_idx * META_DIM * sizeof(int));
            int src_token = meta_read_ptr[2];
            int src_k = meta_read_ptr[3];

            if (src_token >= 0 && src_token < num_tokens && src_k >= 0 && src_k < experts_per_token) {
                s_src_token = src_token;
                s_weight = weights[s_src_token * experts_per_token + src_k];
                s_is_valid = true;
            }
        }
    }
    __syncthreads();

    if (!s_is_valid) return;

    int src_rank = 0;
    #pragma unroll
    for (int i = 1; i < WORLD_SIZE + 1; ++i) if (tid >= s_recv_offsets[i]) src_rank = i;
    int intra_rank_idx = tid - s_recv_offsets[src_rank];
    const __half* data_read_ptr = reinterpret_cast<const __half*>(recv_buf + recv_data_byte_offsets[src_rank] + intra_rank_idx * hidden_dim * sizeof(__half));

    for (int h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        float val = __half2float(data_read_ptr[h]) * s_weight;
        atomicAdd(&out_tokens_f32[s_src_token * hidden_dim + h], val);
    }
}


// C++ wrappers
void hip_count_and_map_dispatches(torch::Tensor i, torch::Tensor sc, torch::Tensor dm, int n, int e, int l) { count_and_map_kernel<<<(n*e+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(i.data_ptr<int>(), sc.data_ptr<int>(), dm.data_ptr<int>(), n, e, l); }
void hip_permute_and_pack(torch::Tensor o, torch::Tensor i, torch::Tensor d, torch::Tensor t, torch::Tensor s, torch::Tensor sdo, torch::Tensor smo, int n, int e, int h, int r, int l) { permute_and_pack_kernel<<<(n*e+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>((const __half*)o.data_ptr<at::Half>(), i.data_ptr<int>(), d.data_ptr<int>(), t.data_ptr<int>(), (unsigned char*)s.data_ptr(), sdo.data_ptr<int>(), smo.data_ptr<int>(), n, e, h, r, l); }
void hip_unpack_and_reorganize(torch::Tensor r, torch::Tensor ex, torch::Tensor em, torch::Tensor en, torch::Tensor rdo, torch::Tensor rmo, torch::Tensor rc, int t, int l, int h, int mr) { unpack_and_reorganize_kernel<<<(t+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>((const unsigned char*)r.data_ptr(), (__half*)ex.data_ptr<at::Half>(), em.data_ptr<int>(), en.data_ptr<int>(), rdo.data_ptr<int>(), rmo.data_ptr<int>(), rc.data_ptr<int>(), t, l, h, mr); }
void hip_count_backward_sends(torch::Tensor em, torch::Tensor en, torch::Tensor sc, int l, int mr) { count_backward_kernel<<<l, BLOCK_SIZE>>>(em.data_ptr<int>(), en.data_ptr<int>(), sc.data_ptr<int>(), l, mr); }
void hip_gather_and_pack(torch::Tensor ey, torch::Tensor em, torch::Tensor en, torch::Tensor t, torch::Tensor s, torch::Tensor sdo, torch::Tensor smo, int l, int mr, int h) { gather_and_pack_kernel<<<l, BLOCK_SIZE>>>((const __half*)ey.data_ptr<at::Half>(), em.data_ptr<int>(), en.data_ptr<int>(), t.data_ptr<int>(), (unsigned char*)s.data_ptr(), sdo.data_ptr<int>(), smo.data_ptr<int>(), l, mr, h); }
void hip_fused_unpack_and_combine(torch::Tensor rb, torch::Tensor w, torch::Tensor otf, torch::Tensor rdo, torch::Tensor rmo, torch::Tensor rc, int t, int h, int e, int n) {
    fused_unpack_and_combine_kernel<<<t, BLOCK_SIZE>>>((const unsigned char*)rb.data_ptr(), w.data_ptr<float>(), otf.data_ptr<float>(), rdo.data_ptr<int>(), rmo.data_ptr<int>(), rc.data_ptr<int>(), t, h, e, n);
}
"""

# 编译HIP模块
hip_module = load_inline(
    name='hip_all2all_async_v4_robust', # 使用新名称避免缓存问题
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=[
        'hip_count_and_map_dispatches', 'hip_permute_and_pack', 'hip_unpack_and_reorganize',
        'hip_count_backward_sends', 'hip_gather_and_pack', 
        'hip_fused_unpack_and_combine'
    ],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20", "-O3"],
)

class HIPAllToAllAsync:
    META_DIM = 5

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size
        
        self.compute_stream = torch.cuda.Stream()

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg
        num_tokens, experts_per_token = indices.shape
        indices_int = indices.to(torch.int32)
        
        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        dispatch_map = torch.empty(num_tokens * experts_per_token, 3, dtype=torch.int32, device=device)

        with torch.cuda.stream(self.compute_stream):
            hip_module.hip_count_and_map_dispatches(indices_int, send_counts, dispatch_map, num_tokens, experts_per_token, self.num_local_experts)

        self.compute_stream.synchronize()
        
        recv_counts_t = torch.empty_like(send_counts, dtype=torch.long)
        dist.all_to_all_single(recv_counts_t, send_counts.to(torch.long))
        recv_counts = recv_counts_t.to(torch.int32)
        total_recv = int(recv_counts.sum().item())

        send_data_bytes = send_counts * cfg.hidden_dim * 2 
        send_meta_bytes = send_counts * self.META_DIM * 4
        send_split_bytes = send_data_bytes + send_meta_bytes
        
        send_buf = torch.empty(send_split_bytes.sum().item(), dtype=torch.uint8, device=device)
        
        with torch.cuda.stream(self.compute_stream):
            if send_split_bytes.sum().item() > 0:
                send_offsets = torch.cumsum(send_split_bytes, 0, dtype=torch.int32) - send_split_bytes
                send_data_offsets = send_offsets
                send_meta_offsets = send_offsets + send_data_bytes
                temp_send_counts = torch.zeros_like(send_counts)
                hip_module.hip_permute_and_pack(dp_x, indices_int, dispatch_map, temp_send_counts, send_buf, send_data_offsets, send_meta_offsets, num_tokens, experts_per_token, cfg.hidden_dim, self.rank, self.num_local_experts)

        recv_data_bytes = recv_counts * cfg.hidden_dim * 2
        recv_meta_bytes = recv_counts * self.META_DIM * 4
        recv_split_bytes = recv_data_bytes + recv_meta_bytes
        recv_buf = torch.empty(recv_split_bytes.sum().item(), dtype=torch.uint8, device=device)

        self.compute_stream.synchronize()
        dist.all_to_all_single(recv_buf, send_buf, recv_split_bytes.tolist(), send_split_bytes.tolist())

        expert_num_tokens = torch.zeros(self.num_local_experts, dtype=torch.int32, device=device)
        expert_x = torch.empty((self.num_local_experts, self.max_recv, cfg.hidden_dim), dtype=cfg.in_dtype, device=device)
        expert_meta = torch.empty((self.num_local_experts, self.max_recv, self.META_DIM), dtype=torch.int32, device=device)

        self.compute_stream.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(self.compute_stream):
            if total_recv > 0:
                recv_offsets = torch.cumsum(recv_split_bytes, 0, dtype=torch.int32) - recv_split_bytes
                recv_data_byte_offsets = recv_offsets
                recv_meta_byte_offsets = recv_offsets + recv_data_bytes
                hip_module.hip_unpack_and_reorganize(recv_buf, expert_x, expert_meta, expert_num_tokens, recv_data_byte_offsets, recv_meta_byte_offsets, recv_counts, total_recv, self.num_local_experts, cfg.hidden_dim, self.max_recv)
        
        self.compute_stream.synchronize()
        return expert_num_tokens, expert_x, expert_meta

    def combine(self, out_tokens: torch.Tensor, weights: torch.Tensor, expert_meta: torch.Tensor, expert_y: torch.Tensor, expert_num_tokens: torch.Tensor, num_tokens: int):
        device = out_tokens.device
        cfg = self.cfg

        send_counts = torch.zeros(self.world_size, dtype=torch.int32, device=device)
        with torch.cuda.stream(self.compute_stream):
            hip_module.hip_count_backward_sends(expert_meta, expert_num_tokens, send_counts, self.num_local_experts, self.max_recv)
        
        self.compute_stream.synchronize()
        recv_counts_t = torch.empty_like(send_counts, dtype=torch.long)
        dist.all_to_all_single(recv_counts_t, send_counts.to(torch.long))
        recv_counts = recv_counts_t.to(torch.int32)
        total_recv = int(recv_counts.sum().item())
        total_sends = int(send_counts.sum().item())
        
        if total_sends > 0:
            send_data_bytes = send_counts * cfg.hidden_dim * 2
            send_meta_bytes = send_counts * self.META_DIM * 4
            send_split_bytes_int = send_data_bytes + send_meta_bytes
            send_split_bytes = send_split_bytes_int.to(torch.long)
            send_buf = torch.empty(send_split_bytes.sum().item(), dtype=torch.uint8, device=device)
            with torch.cuda.stream(self.compute_stream):
                send_offsets = torch.cumsum(send_split_bytes_int, 0, dtype=torch.int32) - send_split_bytes_int
                send_data_offsets = send_offsets
                send_meta_offsets = send_offsets + send_data_bytes
                temp_send_counts = torch.zeros_like(send_counts)
                hip_module.hip_gather_and_pack(expert_y, expert_meta, expert_num_tokens, temp_send_counts, send_buf, send_data_offsets, send_meta_offsets, self.num_local_experts, self.max_recv, cfg.hidden_dim)
        else:
            send_buf = torch.empty(0, dtype=torch.uint8, device=device)
            send_split_bytes = torch.zeros_like(send_counts, dtype=torch.long)

        recv_data_bytes = recv_counts * cfg.hidden_dim * 2
        recv_meta_bytes = recv_counts * self.META_DIM * 4
        recv_split_bytes = recv_data_bytes + recv_meta_bytes
        recv_buf = torch.empty(recv_split_bytes.sum().item(), dtype=torch.uint8, device=device)

        self.compute_stream.synchronize()
        dist.all_to_all_single(recv_buf, send_buf, recv_split_bytes.tolist(), send_split_bytes.tolist())
        
        if total_recv > 0:
            out_tokens_f32 = torch.zeros_like(out_tokens, dtype=torch.float32)
            self.compute_stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(self.compute_stream):
                recv_offsets = torch.cumsum(recv_split_bytes, 0, dtype=torch.int32) - recv_split_bytes
                recv_data_byte_offsets = recv_offsets
                recv_meta_byte_offsets = recv_offsets + recv_data_bytes
                hip_module.hip_fused_unpack_and_combine(recv_buf, weights, out_tokens_f32, recv_data_byte_offsets, recv_meta_byte_offsets, recv_counts, total_recv, cfg.hidden_dim, cfg.experts_per_token, num_tokens)
                out_tokens.copy_(out_tokens_f32)
        
        self.compute_stream.synchronize()
        return out_tokens

# 顶层调用函数
def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    
    hip_all2all = HIPAllToAllAsync(cfg, rank, world_size)
    
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

