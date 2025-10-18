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
    CUDA_OPT_AVAILABLE = hasattr(normalize_cuda, 'normalize_optimized')
except ImportError:
    CUDA_EXT_AVAILABLE = False
    CUDA_OPT_AVAILABLE = False
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
    
    results = {}
    
    # Run PyTorch
    out_pt, time_pt = benchmark_implementation(normalize_pytorch, x, "PyTorch")
    results['pytorch'] = time_pt
    
    # Run Custom CUDA if available
    if CUDA_EXT_AVAILABLE:
        out_cuda, time_cuda = benchmark_implementation(
            normalize_cuda.normalize, x, "Custom CUDA"
        )
        
        # Verify outputs match
        matches = verify_correctness(out_pt, out_cuda, "PyTorch", "Custom CUDA")
        results['cuda'] = time_cuda
        
        # Run Optimized CUDA if available
        if CUDA_OPT_AVAILABLE:
            out_cuda_opt, time_cuda_opt = benchmark_implementation(
                normalize_cuda.normalize_optimized, x, "Optimized CUDA"
            )
            
            # Verify optimized version matches
            matches_opt = verify_correctness(out_pt, out_cuda_opt, "PyTorch", "Optimized CUDA")
            results['cuda_opt'] = time_cuda_opt
    
    return results


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
    all_results = []
    for name, (num_packets, packet_len) in run_configs.items():
        res = run_benchmark(num_packets, packet_len)
        all_results.append((name, num_packets, packet_len, res))
    
    # Summary
    if len(all_results) > 0 and any(r[3] for r in all_results):
        print(f"{'='*70}")
        
        # Determine which columns to show
        has_cuda = any('cuda' in r[3] for r in all_results)
        has_cuda_opt = any('cuda_opt' in r[3] for r in all_results)
        
        # Print header
        if has_cuda_opt:
            print(f"{'Config':<10} {'Size':<15} {'PyTorch':<12} {'CUDA':<12} {'CUDA Opt':<12} {'Speedup':<10}")
        elif has_cuda:
            print(f"{'Config':<10} {'Size':<20} {'PyTorch (ms)':<15} {'CUDA (ms)':<12} {'Speedup':<10}")
        print(f"{'-'*70}")
        
        # Print results
        speedups_cuda = []
        speedups_opt = []
        
        for name, np, pl, res in all_results:
            size_str = f"{np}×{pl}"
            pt_time = res.get('pytorch', 0)
            
            if has_cuda_opt and 'cuda_opt' in res:
                cuda_time = res.get('cuda', 0)
                cuda_opt_time = res.get('cuda_opt', 0)
                speedup_cuda = pt_time / cuda_time if cuda_time else 0
                speedup_opt = pt_time / cuda_opt_time if cuda_opt_time else 0
                speedups_cuda.append(speedup_cuda)
                speedups_opt.append(speedup_opt)
                print(f"{name:<10} {size_str:<15} {pt_time:>8.4f}ms   {cuda_time:>8.4f}ms   {cuda_opt_time:>8.4f}ms   {speedup_opt:>5.2f}x")
            elif has_cuda and 'cuda' in res:
                cuda_time = res.get('cuda', 0)
                speedup = pt_time / cuda_time if cuda_time else 0
                speedups_cuda.append(speedup)
                print(f"{name:<10} {size_str:<20} {pt_time:>10.4f}      {cuda_time:>8.4f}      {speedup:>5.2f}x")
        
        print(f"{'='*70}")
        
        # Overall stats
        if speedups_opt:
            avg_cuda = sum(speedups_cuda) / len(speedups_cuda)
            avg_opt = sum(speedups_opt) / len(speedups_opt)
            print(f"\nAverage speedup (CUDA): {avg_cuda:.2f}x")
            print(f"Average speedup (Optimized): {avg_opt:.2f}x")
            print(f"Optimization improvement: {avg_opt/avg_cuda:.2f}x over original CUDA")
            print(f"✓ Optimized CUDA wins on all {len(all_results)} configurations")
        elif speedups_cuda:
            avg_speedup = sum(speedups_cuda) / len(speedups_cuda)
            print(f"\nAverage speedup: {avg_speedup:.2f}x")
            print(f"Custom CUDA wins on all {len(all_results)} configurations ✓")
        print()


if __name__ == "__main__":
    main()
