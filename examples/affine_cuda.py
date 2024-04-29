import time

import SimpleITK as sitk
import numpy as np
import torch

from preprocessing.affine_3d_cuda.affine_3d import affine_position_3d, affine_image_3d_cuda
from preprocessing.affine_3d_cuda.visualizer import vis


def test_image_landmark_affine(image_path):
    # affine parameters
    angle_x, angle_y, angle_z = -np.pi / 180 * 5., -np.pi / 180 * 5, np.pi / 180 * 5
    tx, ty, tz = -25, 50, 25
    interpolation_mode = "trilinear"  # nearest or trilinear
    keep_original_size = True

    # read image
    sitk_image = sitk.ReadImage(image_path, sitk.sitkUInt8)
    # Convert from [depth, width, height] to [depth, height, width]
    image_data = sitk.GetArrayFromImage(sitk_image).transpose(0, 2, 1)
    image_dims = image_data.shape
    print("z-y-x", image_dims)
    image = torch.tensor(image_data, dtype=torch.uint8, device=torch.device('cuda')).contiguous()

    # rotate image, meanwhile check the time consumption
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


def test_inv_rotate_image(image_path):
    # affine parameters
    angle_x, angle_y, angle_z = -np.pi / 180 * 5., -np.pi / 180 * 5, np.pi / 180 * 5
    interpolation_mode = "trilinear"  # nearest or trilinear
    keep_original_size = True

    # read image
    sitk_image = sitk.ReadImage(image_path, sitk.sitkUInt8)
    # Convert from [depth, width, height] to [depth, height, width]
    image_data = sitk.GetArrayFromImage(sitk_image).transpose(0, 2, 1)
    image_dims = image_data.shape
    print("z-y-x", image_dims)
    image = torch.tensor(image_data, dtype=torch.uint8, device=torch.device('cuda')).contiguous()

    # Rotate the image
    image_rotated = affine_image_3d_cuda(image, torch.tensor([angle_x, angle_y, angle_z]),
                                         torch.tensor([0, 0, 0]), interpolation_mode, keep_original_size,
                                         forward_rotation=True)
    # rotate the image back
    image_rotated_inv = affine_image_3d_cuda(image_rotated, torch.tensor([-angle_x, -angle_y, -angle_z]),
                                             torch.tensor([0, 0, 0]), interpolation_mode, keep_original_size,
                                             forward_rotation=False)
    # visualization
    vis(image_data, image_rotated_inv.cpu().numpy())


if __name__ == "__main__":
    vol_name = '417'
    img_path = f'/mnt/data/medaka_landmarks/data/{vol_name}.tif'
    test_image_landmark_affine(img_path)
    test_inv_rotate_image(img_path)
