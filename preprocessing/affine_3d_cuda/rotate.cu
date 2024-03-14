#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "enums.h"

__device__ float trilinear_interpolation(const u_char *input, int width, int height, int depth,
                                         float x, float y, float z) {
    int x0 = floor(x);
    int y0 = floor(y);
    int z0 = floor(z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height && z0 >= 0 && z1 < depth ){
        float xd = x - x0;
        float yd = y - y0;
        float zd = z - z0;
        // interpolation along x
        float c00 = input[(z0 * height + y0) * width + x0] * (1 - xd) + input[(z0 * height + y0) * width + x1] * xd;
        float c01 = input[(z0 * height + y1) * width + x0] * (1 - xd) + input[(z0 * height + y1) * width + x1] * xd;
        float c10 = input[(z1 * height + y0) * width + x0] * (1 - xd) + input[(z1 * height + y0) * width + x1] * xd;
        float c11 = input[(z1 * height + y1) * width + x0] * (1 - xd) + input[(z1 * height + y1) * width + x1] * xd;
        // interpolation along y
        float c0 = c00 * (1 - yd) + c01 * yd;
        float c1 = c10 * (1 - yd) + c11 * yd;
        // interpolation along z
        return static_cast<u_char>(c0 * (1 - zd) + c1 * zd);
    }else{
        return 0;
    }
}

__global__ void rotate_image_kernel(const u_char *input, u_char *output,
                         int width, int height, int depth,
                         int new_width, int new_height, int new_depth,
                         float new_min_x, float new_min_y, float new_min_z,
                         float angle_x, float angle_y, float angle_z,
                         InterpolationMethod interpolation) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float center_x = width * 0.5;
    float center_y = height * 0.5;
    float center_z = depth * 0.5;

    if (x < new_width && y < new_height && z < new_depth) {// coordinates on output P
    // Find the coordinates on the input P_0
    // P = RzRyRxP0  ---> P0 = inv(Rx)inv(Ry)inv(Rz)P
    float dx = x + new_min_x;
    float dy = y + new_min_y;
    float dz = z + new_min_z;

    // Apply inv(Rz)
    float x_orig = dx * cosf(angle_z) + dy * sinf(angle_z);
    float y_orig = -dx * sinf(angle_z) + dy * cosf(angle_z);
    float z_orig = dz;

    // Apply inv(Ry)
    float tmp_z = z_orig * cosf(angle_y) + x_orig * sinf(angle_y);
    x_orig = -z_orig * sinf(angle_y) + x_orig * cosf(angle_y);
    z_orig = tmp_z;

    // Apply inv(Rx)
    float tmp_y = y_orig * cosf(angle_x) + z_orig * sinf(angle_x);
    z_orig = -y_orig * sinf(angle_x) + z_orig * cosf(angle_x);
    y_orig = tmp_y;

    // Translate back to original coordinates
    x_orig += center_x;
    y_orig += center_y;
    z_orig += center_z;

    if (interpolation==InterpolationMethod::NEAREST){
        // Use nearest-neighbor interpolation
        int x_orig_int = roundf(x_orig);
        int y_orig_int = roundf(y_orig);
        int z_orig_int = roundf(z_orig);
        if (x_orig_int >= 0 && x_orig_int < width &&
            y_orig_int >= 0 && y_orig_int < height &&
            z_orig_int >= 0 && z_orig_int < depth ){
            uint64_t index_output = (static_cast<uint64_t>(z) * new_height + y) * new_width + x;
            uint64_t index_input = (static_cast<uint64_t>(z_orig_int) * height + y_orig_int) * width + x_orig_int;
            output[index_output] = input[index_input];
            }
        }else{
            // Use trilinear interpolation
            uint64_t index_output = (static_cast<uint64_t>(z) * new_height + y) * new_width + x;
            output[index_output] = trilinear_interpolation(input, width, height, depth, x_orig, y_orig, z_orig);
        }
    }
}


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


void launch_rotate_image_kernel(const u_char *input, u_char *output,
                                int width, int height, int depth, int new_width, int new_height, int new_depth,
                                float new_min_x, float new_min_y, float new_min_z,
                                float angle_x, float angle_y, float angle_z,
                                InterpolationMethod interpolation) {
    // Calculate grid dimensions based on input size
    dim3 blockDim(8, 8, 16);
    dim3 gridDim((new_width + blockDim.x - 1) / blockDim.x,
                 (new_height + blockDim.y - 1) / blockDim.y,
                 (new_depth + blockDim.z - 1) / blockDim.z);

    rotate_image_kernel<<<gridDim, blockDim>>>(input, output, width, height, depth, new_width, new_height, new_depth,
                                               new_min_x, new_min_y, new_min_z, angle_x, angle_y, angle_z,
                                               interpolation);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Synchronize to wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

}
