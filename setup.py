from setuptools import setup, find_packages

setup(
    name='preprocessing',
    version='0.0.1',
    packages=find_packages(where="affine_3d_cuda"),
    package_data={'preprocessing': ['affine_3d_cuda/*.so']},
    install_requires=[
        # list your dependencies here
    ],
)
