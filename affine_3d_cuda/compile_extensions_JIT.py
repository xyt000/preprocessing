import os
import shutil

# add python path
my_env = os.environ.copy()
my_env["PATH"] = "/home/ws/ml0077/miniconda3/envs/preprocessing/bin:" + my_env["PATH"]
os.environ.update(my_env)

# set CUDA_HOME
cuda_home = "/home/ws/ml0077/miniconda3/envs/preprocessing/pkgs/cuda-toolkit/"
os.environ["CUDA_HOME"] = cuda_home

# # Compile with TORCH_USE_CUDA_DSA to enable device-side assertions
# os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

import torch.utils.cpp_extension

# Load the C++ extension module
rotate_image_cpp = torch.utils.cpp_extension.load(name='rotate_image_cpp',
                                                  sources=['rotate.cpp', 'rotate.cu'],
                                                  verbose=True,
                                                  extra_cuda_cflags=['-DTORCH_USE_CUDA_DSA'])
print(rotate_image_cpp.__file__)
shutil.move(rotate_image_cpp.__file__, os.getcwd())
