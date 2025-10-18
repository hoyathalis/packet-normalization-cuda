#!/bin/bash
# Per-Packet Normalization Benchmark: PyTorch vs Custom CUDA
# Automatically builds CUDA extension if needed

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment
source /root/miniconda3/bin/activate dvfs

echo "================================================================================"
echo "Per-Packet Normalization: PyTorch vs Custom CUDA Benchmark"
echo "================================================================================"
echo ""

# Check if CUDA extension exists
if [ ! -f "normalize_cuda.cpython-310-aarch64-linux-gnu.so" ]; then
    echo "CUDA extension not found. Building..."
    echo ""
    
    # Clean previous builds
    rm -rf build *.so 2>/dev/null || true
    
    # Build with proper settings
    CUDA_HOME=/usr/local/cuda-12.6 \
    PATH=/usr/local/cuda-12.6/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="9.0+PTX" \
    python setup.py build_ext --inplace > /dev/null 2>&1
    
    if [ -f "normalize_cuda.cpython-310-aarch64-linux-gnu.so" ]; then
        echo "✓ CUDA extension built successfully"
        echo ""
    else
        echo "✗ Failed to build CUDA extension"
        echo "Running with PyTorch only..."
        echo ""
    fi
else
    echo "✓ CUDA extension found"
    echo ""
fi

# Run benchmark
python benchmark.py all

echo ""
echo "================================================================================"
echo "Benchmark Complete!"
echo "================================================================================"

