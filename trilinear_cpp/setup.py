from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if torch.cuda.is_available():
    print('Including CUDA code.')
    setup(
        name='trilinear',
        ext_modules=[
            CUDAExtension('trilinear', [
                'src/trilinear_cuda.cpp',
                'src/trilinear_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found. Error.')
    
