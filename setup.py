from setuptools import setup, find_packages

setup(
    name='preprocessing',
    version='0.0.1',
    packages=find_packages(),
    package_data={'preprocessing': ['affine_3d_cuda/*.os']},
    install_requires=[
        # list your dependencies here
    ],
)
