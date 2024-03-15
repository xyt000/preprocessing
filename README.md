## Preprocessing, Transformation, and Augmentation of 3D images with landmarks for Machine Learning

### Dependencies Installation

To set up the necessary dependencies, create a Conda environment using the provided `conda.yml` file:

```conda env create --name preprocessing --file conda.yml```

### Installing the Preprocessing Package

Install the `preprocessing` package using the following command:

```pip install git+https://github.com/xyt000/preprocessing.git@v0.0.1```


### Usage: Performing Image Affine Transformations

Utilize the `preprocessing.affine` module to perform image affine transformations. Refer to the provided example notebook `examples/affine.ipynb` for implementation details.

**Features**:
- Leveraging Torch functions `F.affine_grid()` and `F.grid_sample()`.
- GPU memory usage concern for images larger than 2GB.

### Recommended: Using `preprocessing.affine_3d_cuda.affine_3d` for Image Affine Transformations

For optimized image affine transformations, we recommend using `preprocessing.affine_3d_cuda.affine_3d`. Refer to the provided example script `examples/affine_cuda.py`.

**Features**:
- Utilizes CUDA, C++, and Torch mixed programming for efficiency.
- Avoids hidden GPU memory allocation issues.
- Approximately 0.025s processing time for 0.5GB image affine transformation.

