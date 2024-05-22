import random

import numpy as np
import torch

from preprocessing.affine_3d_cuda.affine_3d import affine_position_3d, affine_image_3d_cuda
from preprocessing.color_transform import contrast_transform_inplace, brightness_transform_inplace


class RandomAugmentation:
    def __init__(self, seed=None):
        """
        Initializes the RandomAugmentation class with an optional seed for reproducibility.

        Parameters:
        - seed (int, optional): Seed for the random number generator to ensure reproducible results. If None,
                                the random number generator is initialized without a fixed seed. Defaults to None.
        """
        self.rng = random.Random() if seed is None else random.Random(seed)

    def random_augmentation_image_landmarks(self, img, landmarks, img_mean=None, contrast_range=(0.9, 1.1),
                                            brightness_range=(0.95, 1.05), rotation_range=(-5., 5.),
                                            translation_range=(-50., 50.), interpolation_mode='nearest',
                                            keep_original_size=True):
        """
        Applies random augmentations to an image and its corresponding landmarks. The augmentations include
        adjustments to the image's contrast and brightness, as well as random 3D affine transformations
        (rotation and translation) applied to both the image and landmarks.

        Parameters:
        - img (torch.Tensor): The input image tensor with dtype=float.
        - landmarks (torch.Tensor): A tensor containing the 3D landmarks associated with the image, with shape (N, 3).
        - img_mean (float, optional): The mean intensity value of the image used for contrast adjustment.
                                      If None, the mean will be computed. Defaults to None.
        - contrast_range (tuple, optional): A tuple specifying the min and max factors for random contrast adjustment.
                                            Defaults to (0.9, 1.1).
        - brightness_range (tuple, optional): A tuple specifying the min and max factors for random brightness adjustment.
                                              Defaults to (0.95, 1.05).
        - rotation_range (tuple, optional): A tuple specifying the min and max rotation angles in degrees for random
                                            rotation around each axis. Defaults to (-5., 5.).
        - translation_range (tuple, optional): A tuple specifying the min and max translation values in pixels for random
                                               translation along each axis. Defaults to (-50., 50.).
        - interpolation_mode (str, optional): The interpolation mode to use for the affine transformation of the image.
                                              Defaults to 'nearest'.
        - keep_original_size (bool, optional): If True, the output image and landmarks will maintain the original size.
                                               Defaults to True.

        Returns:
        - torch.Tensor: The augmented image tensor.
        - torch.Tensor: The transformed landmarks tensor with the same shape as the input landmarks.
        """
        # Contrast adjustment
        factor = self.rng.uniform(*contrast_range)
        contrast_transform_inplace(img, factor, img_mean)

        # Brightness adjustment
        factor = self.rng.uniform(*brightness_range)
        brightness_transform_inplace(img, factor)

        # Random affine transformations
        angle_x, angle_y, angle_z = self.rng.uniform(*rotation_range), self.rng.uniform(
            *rotation_range), self.rng.uniform(*rotation_range)
        rotation = torch.tensor([np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)]).float()
        tx, ty, tz = self.rng.uniform(*translation_range), self.rng.uniform(
            *translation_range), self.rng.uniform(*translation_range)
        img = affine_image_3d_cuda(img, rotation, torch.tensor([tx, ty, tz]), interpolation_mode, keep_original_size)

        # Adjust landmarks according to the affine transformation
        landmarks = affine_position_3d(img.shape, rotation, torch.tensor([tx, ty, tz]),
                                       landmarks, keep_original_size)
        return img, landmarks

    def random_affine_image(self, img, rotation_range_x=(-5., 5.), rotation_range_y=(-5., 5.),
                            rotation_range_z=(-5., 5.), translate_range_x=(-0., 0.), translate_range_y=(-0., 0.),
                            translate_range_z=(-0., 0.), interpolation_mode='nearest', keep_original_size=True):
        """
        Applies a random affine transformation to a 3D image, including rotations and translations
        within specified ranges for each axis.

        Args:
            img (torch.Tensor): The input image tensor with shape (D, H, W), representing depth, height, and width.
            rotation_range_x (tuple, optional): The range of angles in degrees for random rotation around the X-axis.
                Defaults to (-5., 5.).
            rotation_range_y (tuple, optional): The range of angles in degrees for random rotation around the Y-axis.
                Defaults to (-5., 5.).
            rotation_range_z (tuple, optional): The range of angles in degrees for random rotation around the Z-axis.
                Defaults to (-5., 5.).
            translate_range_x (tuple, optional): The range of translations in the X direction.
                Defaults to (-0., 0.).
            translate_range_y (tuple, optional): The range of translations in the Y direction.
                Defaults to (-0., 0.).
            translate_range_z (tuple, optional): The range of translations in the Z direction.
                Defaults to (-0., 0.).
            interpolation_mode (str, optional): The interpolation mode for resizing the transformed image.
                Can be 'nearest' or 'trilinear'. Defaults to 'nearest'.
            keep_original_size (bool, optional): If True, the output image will be cropped or padded to retain the
                original dimensions. Defaults to True.

        Returns:
            torch.Tensor: The transformed image tensor. The dimensions match the input image if `keep_original_size`
                is True; otherwise, they may vary based on the transformation.
            torch.Tensor: The tensor containing the rotation angles applied to the X, Y, and Z axes, in radians.
            torch.Tensor: The tensor containing the translation values applied to the X, Y, and Z axes.
        """
        # Random rotation angles
        angle_x, angle_y, angle_z = self.rng.uniform(*rotation_range_x), self.rng.uniform(
            *rotation_range_y), self.rng.uniform(*rotation_range_z)
        tx, ty, tz = self.rng.uniform(*translate_range_x), self.rng.uniform(*translate_range_y), self.rng.uniform(
            *translate_range_z)
        rotation = torch.tensor([np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)]).float()
        translation = torch.tensor([tx, ty, tz]).float()

        # Apply 3D rotation
        img = affine_image_3d_cuda(img, rotation, translation, interpolation_mode=interpolation_mode,
                                   keep_original_size=keep_original_size)
        return img, rotation, translation

