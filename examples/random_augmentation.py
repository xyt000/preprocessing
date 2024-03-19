import time

import torch

from preprocessing.affine_3d_cuda.visualizer import vis
from preprocessing.augmentation import random_augmentation_image_landmarks


def main():
    # input image
    vol_name = '417'
    img_path = f'/mnt/data/medaka_landmarks/data/{vol_name}.tif'
    import SimpleITK as sitk
    sitk_image = sitk.ReadImage(img_path, sitk.sitkUInt8)
    # Convert from [depth, width, height] to [depth, height, width]
    image_data = sitk.GetArrayFromImage(sitk_image).transpose(0, 2, 1)
    image_dims = image_data.shape
    print("z-y-x", image_dims)
    image = torch.tensor(image_data, dtype=torch.uint8, device=torch.device('cuda')).contiguous().float()
    torch.cuda.empty_cache()

    for i in range(10):
        # rotate image
        t0 = time.time()
        p = torch.tensor([[231., 265, 600]])
        image_, p_ = random_augmentation_image_landmarks(image, p, seed=42)
        print(f'consumed {time.time() - t0}')

    # visualization
    vis(image.cpu().numpy(), image_.cpu().numpy(), p, p_)


if __name__ == "__main__":
    main()
