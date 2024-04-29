#include <torch/extension.h>
#include "rotate.h"

template<typename T>
void rotate_image_template(const torch::Tensor &input,
                           torch::Tensor &output,
                           float new_min_x, float new_min_y, float new_min_z,
                           float angle_x, float angle_y, float angle_z,
                           InterpolationMethod interpolation=InterpolationMethod::NEAREST, bool forward_rotation=true) {
    // Get input tensor dimensions
    int depth = input.size(0);
    int height = input.size(1);
    int width = input.size(2);

    int new_depth = output.size(0);
    int new_height = output.size(1);
    int new_width = output.size(2);

    // Launch CUDA kernel
    launch_rotate_image_kernel(reinterpret_cast<const T*>(input.data_ptr()),
                               reinterpret_cast<T*>(output.data_ptr()),
                               width, height, depth, new_width, new_height, new_depth,
                               new_min_x, new_min_y, new_min_z, angle_x, angle_y, angle_z,
                               interpolation, forward_rotation);
}


void rotate_image(const torch::Tensor &input,
                  torch::Tensor &output,
                  float new_min_x, float new_min_y, float new_min_z,
                  float angle_x, float angle_y, float angle_z,
                  InterpolationMethod interpolation=InterpolationMethod::NEAREST, bool forward_rotation=true) {
    // Ensure input and output tensors have the same data type
    TORCH_CHECK(input.dtype() == output.dtype(), "Input and output tensors must have the same data type");
    // Ensure input tensor is on CUDA
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    // Ensure input tensor is contiguous
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    // Dispatch based on tensor data type
    if (input.dtype() == torch::kUInt8) {
        rotate_image_template<u_char>(input, output, new_min_x, new_min_y, new_min_z, angle_x, angle_y, angle_z, interpolation, forward_rotation);
    } else if (input.dtype() == torch::kFloat32) {
        rotate_image_template<float>(input, output, new_min_x, new_min_y, new_min_z, angle_x, angle_y, angle_z, interpolation, forward_rotation);
    } else {
        TORCH_CHECK(false, "Unsupported data type. Only uint8 and float32 are supported.");
    }
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
          py::arg("interpolation") = InterpolationMethod::NEAREST, py::arg("forward_rotation") = true);
}
