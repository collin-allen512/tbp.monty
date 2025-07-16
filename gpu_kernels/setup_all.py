from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA detected - building all GPU kernels with CUDA support")
    ext_modules = [
        CUDAExtension(
            name='monty_cuda',
            sources=[
                'monty_cuda_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_70,code=sm_70',  # V100
                    '-gencode', 'arch=compute_75,code=sm_75',  # RTX 2080
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 4070
                ]
            }
        )
    ]
else:
    print("CUDA not available - building CPU-only version")
    ext_modules = [
        Extension(
            name='monty_cuda',
            sources=[
                'monty_cpu_only.cpp',
            ],
            include_dirs=torch.utils.cpp_extension.include_paths(),
            library_dirs=torch.utils.cpp_extension.library_paths(),
            libraries=['torch', 'torch_python'],
            extra_compile_args=['-O3'],
        )
    ]

setup(
    name='monty_cuda',
    version='0.1.0',
    description='CUDA kernels for GPU-accelerated Monty hypothesis updates',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'numpy',
    ],
)