#include <torch/extension.h>
#include "rotate.h"
#include "enums.h"

void rotate_image(const torch::Tensor &input,
                  torch::Tensor &output,
                  float new_min_x, float new_min_y, float new_min_z,
                  float angle_x, float angle_y, float angle_z,
                  InterpolationMethod interpolation=InterpolationMethod::NEAREST) {
    // Get input tensor dimensions
    int depth = input.size(0);
    int height = input.size(1);
    int width = input.size(2);

    int new_depth = output.size(0);
    int new_height = output.size(1);
    int new_width = output.size(2);

    // Launch CUDA kernel
    launch_rotate_image_kernel((const u_char *)input.data_ptr(), (u_char *)output.data_ptr(),
                               width, height, depth, new_width, new_height, new_depth,
                               new_min_x, new_min_y, new_min_z, angle_x, angle_y, angle_z,
                               interpolation);
}

// Binding the rotate_image function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Define the InterpolationMethod enum in Python
    py::enum_<InterpolationMethod>(m, "InterpolationMethod")
        .value("NEAREST", InterpolationMethod::NEAREST)
        .value("TRILINEAR", InterpolationMethod::TRILINEAR)
        .export_values();

    // Bind the rotate_image function with default arguments
    m.def("rotate_image", &rotate_image, "Rotate a 3D image tensor using CUDA",
          py::arg("input"), py::arg("output"), py::arg("new_min_x"), py::arg("new_min_y"), py::arg("new_min_z"),
          py::arg("angle_x"), py::arg("angle_y"), py::arg("angle_z"),
          py::arg("interpolation") = InterpolationMethod::NEAREST);
}