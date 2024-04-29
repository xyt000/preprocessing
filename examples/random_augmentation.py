import time

import numpy as np
import torch
import SimpleITK as sitk
import torch.nn.functional as F


from preprocessing.affine_3d_cuda.visualizer import vis
from preprocessing.augmentation import RandomAugmentation


def test_aug_image_with_landmarks(image_path):
    # initialize the RandomAugmentation with seed
    ra = RandomAugmentation(seed=42)
    # read image
    sitk_image = sitk.ReadImage(image_path, sitk.sitkUInt8)
    # Convert from [depth, width, height] to [depth, height, width]
    image_data = sitk.GetArrayFromImage(sitk_image).transpose(0, 2, 1)
    image_dims = image_data.shape
    print("z-y-x", image_dims)
    image = torch.tensor(image_data, dtype=torch.uint8, device=torch.device('cuda')).contiguous().float()
    torch.cuda.empty_cache()
    # apply random augmentation on the image with annotated landmarks
    for i in range(10):
        # augment image and landmarks
        t0 = time.time()
        p = torch.tensor([[231., 265, 600]])  # landmarks (x-y-z)
        image_, p_ = ra.random_augmentation_image_landmarks(image, p)
        print(f'consumed {time.time() - t0}')

    # visualization
    vis(image_data, image_.cpu().numpy(), p, p_)


def test_random_rotation(image_path):
    ra = RandomAugmentation(seed=42)
    # read image
    sitk_image = sitk.ReadImage(img_path, sitk.sitkUInt8)
    # Convert from [depth, width, height] to [depth, height, width]
    image_data = sitk.GetArrayFromImage(sitk_image).transpose(0, 2, 1)
    image_dims = image_data.shape
    print("z-y-x", image_dims)

    # down sampling
    avg_pool = torch.nn.AvgPool3d(kernel_size=5, stride=5).to('cuda')
    img = torch.from_numpy(image_data).pin_memory().to('cuda').float().unsqueeze(0)
    img = torch.round(avg_pool(img).squeeze(0)).type(torch.uint8).contiguous().float()
    # pad image to target size
    d, h, w = [tg - s for tg, s in zip((576, 175, 202), img.shape)]
    img = F.pad(img, [w // 2, w - w // 2, h // 2, h - h // 2, d // 2, d - d // 2])
    torch.cuda.empty_cache()

    # random rotate image
    ts = []
    for i in range(10):
        # augment image and landmarks
        tik = time.time()
        image_, rotation = ra.random_rotate_image(img, rotation_range_x=[20., 20.], rotation_range_y=[0., 0.],
                                                  rotation_range_z=[-0., 0.])
        print(f'consumed {time.time() - tik}')

    # visualization
    vis(image_data, image_.cpu().numpy(), None, None)


if __name__ == "__main__":
    vol_name = '417'
    img_path = f'/mnt/data/medaka_landmarks/data/{vol_name}.tif'
    test_aug_image_with_landmarks(img_path)
    test_random_rotation(img_path)
