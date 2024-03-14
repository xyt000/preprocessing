import time

import numpy as np
import torch

from affine_3d_cuda.affine_3d import affine_position_3d, affine_image_3d_cuda
from affine_3d_cuda.visualizer import vis


def main():
    # affine parameters
    angle_x, angle_y, angle_z = -np.pi / 180 * 5., -np.pi / 180 * 5, np.pi / 180 * 5
    tx, ty, tz = -25, 50, 25
    interpolation_mode = "trilinear"  # nearest or trilinear
    keep_original_size = True

    # input image
    vol_name = '417'
    img_path = f'/mnt/data/medaka_landmarks/data/{vol_name}.tif'
    import SimpleITK as sitk
    sitk_image = sitk.ReadImage(img_path, sitk.sitkUInt8)
    # Convert from [depth, width, height] to [depth, height, width]
    image_data = sitk.GetArrayFromImage(sitk_image).transpose(0, 2, 1)
    image_dims = image_data.shape
    print("z-y-x", image_dims)
    image = torch.tensor(image_data, dtype=torch.uint8, device=torch.device('cuda')).contiguous()

    for i in range(10):
        # rotate image
        t0 = time.time()
        # Rotate the input tensor
        image_rotated = affine_image_3d_cuda(image, torch.tensor([angle_x, angle_y, angle_z]),
                                             torch.tensor([tx, ty, tz]), interpolation_mode, keep_original_size)
        print(f'consumed {time.time() - t0}')

    # Print the shapes of the input and output tensors
    print("Input tensor shape:", image.shape)
    print("Output tensor shape:", image_rotated.shape)

    # landmark (vert1)
    p = torch.tensor([[231., 265, 600]])
    p_transformed = affine_position_3d(image_dims, torch.tensor([angle_x, angle_y, angle_z]),
                                       torch.tensor([tx, ty, tz]), p, keep_original_size)
    # visualization
    vis(image.cpu().numpy(), image_rotated.cpu().numpy(), p, p_transformed)


if __name__ == "__main__":
    main()
