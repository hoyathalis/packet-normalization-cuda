"""
Setup script to build the custom CUDA extension for normalization.

Build with:
    python setup.py build_ext --inplace
    
Or to skip CUDA version check (if you know what you're doing):
    TORCH_CUDA_ARCH_LIST="9.0" python setup.py build_ext --inplace
"""
import os
import warnings
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Set CUDA_HOME if not already set
if 'CUDA_HOME' not in os.environ:
    # Try to find CUDA installation
    for cuda_path in ['/usr/local/cuda-12.6', '/usr/local/cuda-13.0', '/usr/local/cuda']:
        if os.path.exists(cuda_path):
            os.environ['CUDA_HOME'] = cuda_path
            print(f"Using CUDA_HOME: {cuda_path}")
            break

# Set explicit CUDA architecture
# GB200 is compute capability 10.0, H100 is 9.0
if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    print(f"Using TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")

setup(
    name='normalize_cuda',
    ext_modules=[
        CUDAExtension(
            'normalize_cuda', 
            [
                'src/normalize_cuda.cpp',
                'src/normalize_cuda_kernel.cu',
                'src/normalize_cuda_kernel_optimized.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
