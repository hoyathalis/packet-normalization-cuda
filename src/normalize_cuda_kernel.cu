#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int num_packets,
    int packet_len
) {
    int packet_idx = blockIdx.x;
    if (packet_idx >= num_packets) return;
    
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_sq_sum = &shared[blockDim.x];
    
    // Step 1: Compute mean via parallel reduction
    float local_sum = 0.0f;
    for (int i = tid; i < packet_len; i += blockDim.x) {
        local_sum += x[packet_idx * packet_len + i];
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction tree
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / fmaxf(packet_len, 1.0f);  // Prevent division by zero
    __syncthreads();
    
    // Step 2: Compute variance via parallel reduction
    float local_sq_sum = 0.0f;
    for (int i = tid; i < packet_len; i += blockDim.x) {
        float diff = x[packet_idx * packet_len + i] - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    __syncthreads();
    
    // Reduction tree for variance
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_sq_sum[0] / fmaxf(packet_len, 1.0f);  // Prevent division by zero
    float std = sqrtf(variance) + 1e-5f;  // Add epsilon for numerical stability
    __syncthreads();
    
    // Step 3: Normalize
    for (int i = tid; i < packet_len; i += blockDim.x) {
        out[packet_idx * packet_len + i] = 
            (x[packet_idx * packet_len + i] - mean) / std;
    }
}

torch::Tensor normalize_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    
    auto out = torch::empty_like(x);
    int num_packets = x.size(0);
    int packet_len = x.size(1);
    
    if (num_packets == 0 || packet_len == 0) {
        return out;  // Return empty tensor for safety
    }
    
    int threads = 256;
    int blocks = num_packets;
    int shared_mem = 2 * threads * sizeof(float);
    
    normalize_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_packets,
        packet_len
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return out;
}
