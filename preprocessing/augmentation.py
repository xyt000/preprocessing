import random

import numpy as np
import torch

from preprocessing.affine_3d_cuda.affine_3d import affine_position_3d, affine_image_3d_cuda
from preprocessing.color_transform import contrast_transform_inplace, brightness_transform_inplace


def random_augmentation_image_landmarks(img, landmarks, img_mean=None, contrast_range=(0.9, 1.1),
                                        brightness_range=(0.95, 1.05), rotation_range=(-5., 5.),
                                        translation_range=(-50., 50.), seed=None, interpolation_mode='nearest',
                                        keep_original_size=True):
    """
    Apply random augmentation to an image and its associated landmarks.

    This function performs a series of random transformations on the input image and its landmarks.
    The transformations include contrast adjustment, brightness adjustment, rotation, and translation.

    Args:
        img (torch.Tensor, float): Input image tensor.
        landmarks (torch.Tensor, float, Nx3): Landmarks associated with the image.
        img_mean (float, optional): Mean value of the image. If None, it will be calculated from the image.
        contrast_range (tuple, optional): Range for contrast adjustment, default is (0.9, 1.1).
        brightness_range (tuple, optional): Range for brightness adjustment, default is (0.95, 1.05).
        rotation_range (tuple, optional): Range for rotation in degrees, default is (-5., 5.).
        translation_range (tuple, optional): Range for translation in pixels, default is (-50., 50.).
        seed (int, optional): Seed for random number generator.
        interpolation_mode (str, optional): Interpolation mode for affine transformation, default is 'nearest'.
        keep_original_size (bool, optional): Whether to keep the original size after transformation, default is True.

    Returns:
        tuple: Tuple containing transformed image tensor and transformed landmarks tensor.

    """
    rng = random.Random() if seed is None else random.Random(seed)

    # Contrast adjustment
    factor = rng.uniform(*contrast_range)
    contrast_transform_inplace(img, factor, img_mean)

    # Brightness adjustment
    factor = rng.uniform(*brightness_range)
    brightness_transform_inplace(img, factor)

    # Random affine
    angle_x, angle_y, angle_z = rng.uniform(*rotation_range), rng.uniform(*rotation_range), rng.uniform(*rotation_range)
    rotation = torch.tensor([np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)]).float()
    tx, ty, tz = rng.uniform(*translation_range), rng.uniform(*translation_range), rng.uniform(*translation_range)
    img = affine_image_3d_cuda(img, rotation, torch.tensor([tx, ty, tz]), interpolation_mode, keep_original_size)

    # Adjust landmarks
    landmarks = affine_position_3d(img.shape, rotation, torch.tensor([tx, ty, tz]),
                                   landmarks, keep_original_size)
    return img, landmarks
