#include "enums.h"
void launch_rotate_image_kernel(const u_char *input, u_char *output,
                                int width, int height, int depth, int new_width, int new_height, int new_depth,
                                float new_min_x, float new_min_y, float new_min_z,
                                float angle_x, float angle_y, float angle_z,
                                InterpolationMethod interpolation);