#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle instructions (faster than shared memory)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized kernel using:
// 1. Warp shuffle reductions (no shared memory for small reductions)
// 2. Welford's online algorithm (single-pass mean+variance)
// 3. Vectorized loads (float4 when aligned)
__global__ void normalize_kernel_optimized(
    const float* __restrict__ x,
    float* __restrict__ out,
    int num_packets,
    int packet_len
) {
    int packet_idx = blockIdx.x;
    if (packet_idx >= num_packets) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    const float* packet_in = x + packet_idx * packet_len;
    float* packet_out = out + packet_idx * packet_len;
    
    // Shared memory for inter-warp reduction only
    __shared__ float shared_mean[8];  // Max 8 warps (256 threads)
    __shared__ float shared_m2[8];
    
    // Welford's online algorithm for numerically stable mean+variance
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    float count = 0.0f;
    float mean = 0.0f;
    float m2 = 0.0f;  // Sum of squared differences
    
    // Process elements with stride
    for (int i = tid; i < packet_len; i += blockDim.x) {
        float val = packet_in[i];
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    // Warp-level reduction
    count = warp_reduce_sum(count);
    mean = warp_reduce_sum(mean);
    m2 = warp_reduce_sum(m2);
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared_mean[warp_id] = mean;
        shared_m2[warp_id] = m2;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp participates)
    if (tid < 32) {
        int num_warps = (blockDim.x + 31) / 32;
        float warp_mean = (tid < num_warps) ? shared_mean[tid] : 0.0f;
        float warp_m2 = (tid < num_warps) ? shared_m2[tid] : 0.0f;
        
        // Reduce across warps
        warp_mean = warp_reduce_sum(warp_mean) / num_warps;
        warp_m2 = warp_reduce_sum(warp_m2);
        
        if (tid == 0) {
            shared_mean[0] = warp_mean;
            shared_m2[0] = warp_m2;
        }
    }
    __syncthreads();
    
    // All threads read final mean and std
    float final_mean = shared_mean[0];
    float variance = shared_m2[0] / fmaxf(packet_len, 1.0f);
    float std = sqrtf(variance) + 1e-5f;
    
    // Normalize with vectorized writes where possible
    int vec_len = packet_len / 4;
    
    // Vectorized path (4 elements at a time)
    for (int i = tid; i < vec_len; i += blockDim.x) {
        int idx = i * 4;
        float4 vals = reinterpret_cast<const float4*>(packet_in)[i];
        vals.x = (vals.x - final_mean) / std;
        vals.y = (vals.y - final_mean) / std;
        vals.z = (vals.z - final_mean) / std;
        vals.w = (vals.w - final_mean) / std;
        reinterpret_cast<float4*>(packet_out)[i] = vals;
    }
    
    // Handle remainder
    for (int i = vec_len * 4 + tid; i < packet_len; i += blockDim.x) {
        packet_out[i] = (packet_in[i] - final_mean) / std;
    }
}

torch::Tensor normalize_cuda_optimized(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    
    auto out = torch::empty_like(x);
    int num_packets = x.size(0);
    int packet_len = x.size(1);
    
    if (num_packets == 0 || packet_len == 0) {
        return out;
    }
    
    // Tune block size based on packet length
    int threads = 256;  // Good balance for most cases
    int blocks = num_packets;
    int shared_mem = 8 * 2 * sizeof(float);  // 8 warps × 2 values
    
    normalize_kernel_optimized<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_packets,
        packet_len
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return out;
}

