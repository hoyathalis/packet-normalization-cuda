# Network Packet Normalization for Anomaly Detection

High-performance CUDA kernel for real-time network packet preprocessing in intrusion detection systems.

## Problem

Network security systems process millions of packets per second. Each packet must be normalized before ML inference:

```
Input:  Raw packet features [batch_size × num_features]
Output: Normalized packets with mean=0, std=1 per packet
```

**Challenge**: At 10 Gbps, we process **150,000 packets/sec**. PyTorch's normalization is too slow for real-time processing.

## Solution

Custom CUDA kernel with **kernel fusion** - computes mean, std, and normalization in a single GPU pass instead of PyTorch's 3 separate kernels.

**Result**: 2-8x faster, handles real-time traffic with sub-millisecond latency.

## Benchmark Results (NVIDIA GB200)

| Batch Size | Features | PyTorch (ms) | CUDA (ms) | Speedup |
|------------|----------|--------------|-----------|---------|
| 1,024 | 128 | 0.0645 | 0.0085 | **7.63x** |
| 4,096 | 256 | 0.0626 | 0.0180 | **3.48x** |
| 8,192 | 512 | 0.0663 | 0.0330 | **2.01x** |
| 16,384 | 1,024 | 0.1873 | 0.0882 | **2.12x** |

Average speedup: **3.81x**

## Roofline Analysis

![Roofline Plot](roofline.png)

Our kernel is **memory bound** (arithmetic intensity = 0.75 FLOPs/byte), achieving **67-101% memory bandwidth utilization**. This confirms kernel fusion is the critical optimization - reducing memory traffic from 6 operations (PyTorch) to 2 (our kernel).

## Quick Start

```bash
# Run benchmark (auto-builds CUDA extension if needed)
./run_benchmark.sh

# Generate roofline analysis
python profile_kernel.py
```

## Project Structure

```
├── src/
│   ├── normalize_pytorch.py      # PyTorch baseline
│   ├── normalize_cuda_kernel.cu  # Fused CUDA kernel
│   └── normalize_cuda.cpp        # Python binding
├── benchmark.py                   # Performance comparison
├── profile_kernel.py             # Roofline analysis
├── run_benchmark.sh              # Main entry point
└── setup.py                       # Build script
```

## Why Custom CUDA?

**PyTorch approach** (slow):
```
1. Read data → compute mean → write mean
2. Read data → compute std → write std  
3. Read data → normalize → write output
Total: 6 memory operations
```

**Our approach** (fast):
```
1. Read data → compute mean+std+normalize → write output
Total: 2 memory operations (3x reduction)
```

GPU performance is limited by memory bandwidth, not compute. Fewer memory operations = faster execution.

## Why Not Numba?

Numba CUDA is easier to use but doesn't work here:
- GB200 has compute capability **10.0** (Blackwell architecture)
- Numba currently supports only up to **9.x** (H100 and earlier)
- Results in segmentation faults on GB200

Custom C++/CUDA works on all GPU generations.

## Requirements

- CUDA-capable GPU (compute capability 9.0+)
- PyTorch with CUDA support
- CUDA Toolkit 12.6+
- Python 3.10+
- matplotlib (for roofline plots)

## Real-World Impact

For a 10 Gbps network processing 1.5M packets/sec in batches of 8K:

```
Batches per second: 1,500,000 / 8,192 = 183 batches/sec

PyTorch: 183 × 0.0663ms = 12.1 ms/sec GPU time
Custom:  183 × 0.0330ms =  6.0 ms/sec GPU time

Savings: 6.1 ms/sec freed for ML model inference
```

This allows processing higher traffic rates or running more complex models on the same hardware.

## Algorithm

Each packet normalized independently:
```
x_norm[i,j] = (x[i,j] - mean(x[i,:])) / (std(x[i,:]) + ε)
```
where ε = 1e-5 prevents division by zero.

## CUDA Kernel Design

- **1 thread block per packet** for independent processing
- **256 threads per block** for parallel reduction
- **Shared memory** for fast mean/std computation
- **Coalesced memory access** for optimal bandwidth
- **Single kernel launch** instead of 3 separate launches

## License

MIT
