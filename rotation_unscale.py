import torch

def rotate_image(image, angle_rad):
    """
    Rotate a 2D image tensor with bilinear interpolation.

    Args:
    - image (torch.Tensor): Input 2D image tensor of shape (height, width)
    - angle (float): Rotation angle in radians

    Returns:
    - rotated_image (torch.Tensor): Rotated 2D image tensor
    """

    # Get image size
    height, width = image.shape

    # Calculate rotation center
    center_x = width / 2
    center_y = height / 2

    # Define rotation matrix
    rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)],
                                    [torch.sin(angle_rad), torch.cos(angle_rad)]])

    # Calculate coordinates after rotation
    new_coords_x = torch.tensor((0, width-1)) - center_x
    new_coords_y = torch.tensor((0, height-1)) - center_y
    new_coords_x, new_coords_y = torch.meshgrid(new_coords_x, new_coords_y)
    new_coords = torch.stack([new_coords_x.flatten(), new_coords_y.flatten()], dim=0)
    rotated_coords = torch.mm(rotation_matrix, new_coords).round()

    # Calculate new image size
    new_width = int(rotated_coords[0].max() - rotated_coords[0].min() + 1)
    new_height = int(rotated_coords[1].max() - rotated_coords[1].min() + 1)

    # Create empty rotated image
    rotated_image = torch.zeros(new_height, new_width, dtype=image.dtype)

    # Calculate translation for filling the rotated image
    translation_x = -rotated_coords[0].min()
    translation_y = -rotated_coords[1].min()

    # Fill rotated image with bilinear interpolation
    for y in range(new_height):
        for x in range(new_width):
            # Calculate original coordinates after inverse rotation and translation
            original_coords = torch.mm(torch.inverse(rotation_matrix),
                                       torch.tensor([[x - translation_x], [y - translation_y]]))
            orig_x, orig_y = original_coords[0, 0].item()+center_x, original_coords[1, 0].item()+center_y

            # Perform bilinear interpolation if original coordinates are within bounds
            if orig_x >= 0 and orig_x < width - 1 and orig_y >= 0 and orig_y < height - 1:
                x0 = int(orig_x)
                y0 = int(orig_y)
                x1 = x0 + 1
                y1 = y0 + 1
                dx = orig_x - x0
                dy = orig_y - y0
                top = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
                bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
                rotated_image[y, x] = top * (1 - dy) + bottom * dy

    return rotated_image

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Example usage
image = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])  # Example 2D image tensor

print(image.shape)
angle = torch.tensor(np.pi/2)  # Rotation angle in radians (90 degrees)

rotated_image = rotate_image(image, angle)
print(rotated_image)


# Load the image
image_path = "./vis/0.png"
image = Image.open(image_path)

# Convert image to PyTorch tensor
image_tensor = transforms.ToTensor()(image)

# Rotate the image by 90 degrees
rotated_image_tensor = rotate_image(image_tensor[0], torch.tensor(np.pi/2))

# Convert rotated image tensor to PIL image
rotated_image_pil = transforms.ToPILImage()(rotated_image_tensor)

# Visualize the original and rotated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(rotated_image_pil)
plt.title('Rotated Image')
plt.axis('off')
plt.show()
