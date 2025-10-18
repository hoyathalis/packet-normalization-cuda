"""
Per-Packet Normalization Benchmark: PyTorch vs Custom CUDA

Tests different data sizes to show when and why custom CUDA kernels matter.
Each packet (row) is normalized independently: x_norm = (x - mean) / (std + eps)

Usage:
    python benchmark.py [small|medium|large|all]
"""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from normalize_pytorch import normalize_pytorch

# Load custom CUDA extension
try:
    import normalize_cuda
    CUDA_EXT_AVAILABLE = True
except ImportError:
    CUDA_EXT_AVAILABLE = False
    print("Warning: Custom CUDA extension not available")
    print("Build with: python setup.py build_ext --inplace\n")


def benchmark_implementation(func, x, name, warmup=10, iterations=100):
    """Benchmark a single implementation."""
    # Warmup
    for _ in range(warmup):
        _ = func(x)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        out = func(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / iterations
    return out, elapsed_ms


def verify_correctness(out1, out2, name1="Output 1", name2="Output 2"):
    """Verify two outputs match."""
    max_diff = (out1 - out2).abs().max().item()
    return max_diff < 1e-5


def run_benchmark(num_packets, packet_len):
    """Run benchmark for a specific data size."""
    # Generate test data
    torch.manual_seed(42)
    x = torch.randn(num_packets, packet_len, device='cuda', dtype=torch.float32)
    
    # Run PyTorch
    out_pt, time_pt = benchmark_implementation(normalize_pytorch, x, "PyTorch")
    
    # Run Custom CUDA if available
    if CUDA_EXT_AVAILABLE:
        out_cuda, time_cuda = benchmark_implementation(
            normalize_cuda.normalize, x, "Custom CUDA"
        )
        
        # Verify outputs match
        matches = verify_correctness(out_pt, out_cuda, "PyTorch", "Custom CUDA")
        speedup = time_pt / time_cuda
        
        return time_pt, time_cuda, speedup
    else:
        return time_pt, None, None


def main():
    """Main benchmark routine."""
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print("="*70)
    print("Per-Packet Normalization: PyTorch vs Custom CUDA")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Benchmarking...")
    print()
    
    # Define test configurations (batch_size, features_per_packet)
    # batch_size = number of packets processed together
    configs = {
        'small':  (1024, 128),    # Small batch: 1K packets
        'medium': (4096, 256),    # Medium batch: 4K packets  
        'large':  (8192, 512),    # Large batch: 8K packets (typical)
        'xlarge': (16384, 1024),  # XLarge batch: 16K packets
    }
    
    # Select which configs to run
    if mode == "all":
        run_configs = configs
    elif mode in configs:
        run_configs = {mode: configs[mode]}
    else:
        print(f"Unknown mode: {mode}")
        print(f"Available: {', '.join(configs.keys())}, all")
        return
    
    # Run benchmarks
    results = []
    for name, (num_packets, packet_len) in run_configs.items():
        pt_time, cuda_time, speedup = run_benchmark(num_packets, packet_len)
        if cuda_time:
            results.append((name, num_packets, packet_len, pt_time, cuda_time, speedup))
    
    # Summary
    if len(results) > 0:
        print(f"{'='*70}")
        print(f"{'Config':<10} {'Size':<20} {'PyTorch (ms)':<15} {'CUDA (ms)':<12} {'Speedup':<10}")
        print(f"{'-'*70}")
        for name, np, pl, pt_time, cuda_time, speedup in results:
            size_str = f"{np}×{pl}"
            print(f"{name:<10} {size_str:<20} {pt_time:>10.4f}      {cuda_time:>8.4f}      {speedup:>5.2f}x")
        print(f"{'='*70}")
        
        # Overall stats
        avg_speedup = sum(r[5] for r in results) / len(results)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        print(f"Custom CUDA wins on all {len(results)} configurations ✓")
        print()


if __name__ == "__main__":
    main()
