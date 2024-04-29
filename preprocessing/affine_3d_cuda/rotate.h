#pragma once

#include "enums.h"

template <typename scalar_t>
void launch_rotate_image_kernel(const scalar_t *input, scalar_t *output,
                                int width, int height, int depth, int new_width, int new_height, int new_depth,
                                float new_min_x, float new_min_y, float new_min_z,
                                float angle_x, float angle_y, float angle_z,
                                InterpolationMethod interpolation, bool forward_rotation);