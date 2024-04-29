from setuptools import setup, find_packages

setup(
    name='preprocessing',
    version='0.0.3',
    packages=find_packages(),
    package_data={'preprocessing.affine_3d_cuda': ['rotate_image_cpp.so']},
    install_requires=[
        # list your dependencies here
    ],
)
