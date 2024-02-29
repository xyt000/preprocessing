import torch
import torch.nn.functional as F


def affine_3d(image, angles, translations, mode='trilinear'):
    """
    Rotate a 3D volumetric image tensor without scaling effects.

    Args:
    - image (torch.Tensor, float): Input 3D volumetric image tensor of shape (depth (z), height (y), width (x))
    - angles (torch.Tensor, float): Rotation angles around the x, y, z -axis in radians, shape (3,), anti-clockwise
    - translations (torch.Tensor, float): translation distances along the x, y, z -axis in pixels, shape (3,)
    - mode (str): 'trilinear' or 'nearest'

    Returns:
    - rotated_image (torch.Tensor, float): Rotated 3D volumetric image tensor
    """

    # Get image size
    depth, height, width = image.shape

    # Calculate rotation center
    center_x = width / 2
    center_y = height / 2
    center_z = depth / 2

    # Define rotation matrices
    angle_x, angle_y, angle_z = angles
    rotation_matrix_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), -torch.sin(angle_x)],
        [0, torch.sin(angle_x), torch.cos(angle_x)]
    ])

    rotation_matrix_y = torch.tensor([
        [torch.cos(angle_y), 0, torch.sin(angle_y)],
        [0, 1, 0],
        [-torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])

    rotation_matrix_z = torch.tensor([
        [torch.cos(angle_z), -torch.sin(angle_z), 0],
        [torch.sin(angle_z), torch.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Calculate coordinates of vertices after rotation
    new_coords_x = torch.tensor((0, width - 1)) - center_x
    new_coords_y = torch.tensor((0, height - 1)) - center_y
    new_coords_z = torch.tensor((0, depth - 1)) - center_z
    new_coords_x, new_coords_y, new_coords_z = torch.meshgrid(new_coords_x, new_coords_y, new_coords_z)
    new_coords = torch.stack([new_coords_x.flatten(), new_coords_y.flatten(), new_coords_z.flatten()], dim=0)

    rotated_coords = torch.mm(rotation_matrix_x, new_coords)
    rotated_coords = torch.mm(rotation_matrix_y, rotated_coords)
    rotated_coords = torch.mm(rotation_matrix_z, rotated_coords)

    # Calculate new image size
    new_width = int(rotated_coords[0].max() - rotated_coords[0].min() + 1)
    new_height = int(rotated_coords[1].max() - rotated_coords[1].min() + 1)
    new_depth = int(rotated_coords[2].max() - rotated_coords[2].min() + 1)

    # scale translations: in F.affine_grid and F.grid_sample the coordinates are scaled to [-1, 1]
    unit_step = torch.tensor([2. / (itm - 1) for itm in [width, height, depth]], device=translations.device)
    translations = translations * unit_step

    # affine matrix
    rotation_matrix = torch.mm(rotation_matrix_z, torch.mm(rotation_matrix_y, rotation_matrix_x))
    rotation_matrix = rotation_matrix * torch.tensor([[new_width / width, new_height / width, new_depth / width],
                                                      [new_width / height, new_height / height, new_depth / height],
                                                      [new_width / depth, new_height / depth, new_depth / depth]])

    affine_matrix = torch.cat([rotation_matrix, translations.T.unsqueeze(-1)], dim=1).to(image.device)

    # Generate grid of coordinates
    grid = F.affine_grid(affine_matrix.unsqueeze(0),
                         torch.Size((1, 1, new_depth, new_height, new_width), device=image.device),
                         align_corners=True)

    # Perform grid sample using trilinear interpolation
    mode = 'bilinear' if mode == 'trilinear' else mode
    rotated_image = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, align_corners=True, mode=mode)
    return rotated_image.squeeze()


def affine_3d_idx(image_size, angles, translations, indexes):
    """
    Calculate the corresponding indexes in the rotated image from indexes in the original image.

    Args:
    - image_size (tuple): Size of the original image (depth(z), height(y), width(x))
    - angles (torch.Tensor): Rotation angles around the x, y, z-axis in radians, shape (3,), anti-clockwise
    - translations (torch.Tensor): Translation distances along the x, y, z-axis in pixels, shape (3,)
    - indexes (torch.Tensor): Indexes (width, height, depth) in the original image, shape (N, 3)

    Returns:
    - rotated_indexes (torch.Tensor): Corresponding indexes (width, height, depth) in the rotated image, shape (N, 3)
    """

    # Get image size
    depth, height, width = image_size

    # Calculate rotation center
    center_x = width / 2
    center_y = height / 2
    center_z = depth / 2

    # Define rotation matrices
    angle_x, angle_y, angle_z = angles * -1
    rotation_matrix_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), -torch.sin(angle_x)],
        [0, torch.sin(angle_x), torch.cos(angle_x)]
    ])

    rotation_matrix_y = torch.tensor([
        [torch.cos(angle_y), 0, torch.sin(angle_y)],
        [0, 1, 0],
        [-torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])

    rotation_matrix_z = torch.tensor([
        [torch.cos(angle_z), -torch.sin(angle_z), 0],
        [torch.sin(angle_z), torch.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Calculate coordinates of vertices after rotation
    new_coords_x = torch.tensor((0, width - 1)) - center_x
    new_coords_y = torch.tensor((0, height - 1)) - center_y
    new_coords_z = torch.tensor((0, depth - 1)) - center_z
    new_coords_x, new_coords_y, new_coords_z = torch.meshgrid(new_coords_x, new_coords_y, new_coords_z)
    new_coords = torch.stack([new_coords_x.flatten(), new_coords_y.flatten(), new_coords_z.flatten()], dim=0)

    # Calculate rotation matrix
    rotation_matrix = torch.mm(rotation_matrix_z, torch.mm(rotation_matrix_y, rotation_matrix_x))
    rotated_coords = torch.mm(rotation_matrix, new_coords)

    # Apply translation and rotation to indexes
    indexes = indexes - torch.tensor([center_x, center_y, center_z]) + translations * -1
    rotated_indexes = torch.mm(rotation_matrix, indexes.float().T)

    # Calculate the minimum coordinates after rotation
    min_coords = torch.tensor([rotated_coords[0].min(), rotated_coords[1].min(), rotated_coords[2].min()])
    rotated_indexes = rotated_indexes.T - min_coords

    return rotated_indexes.round()


def rotate_image_2d(image, angle):
    """
    Rotate a 2D image tensor without scaling effects.

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
    aspect_ratio = width / height

    # Define rotation matrix
    rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                                    [torch.sin(angle), torch.cos(angle)]])

    # Calculate coordinates after rotation
    new_coords_x = torch.tensor((0, width - 1)) - center_x
    new_coords_y = torch.tensor((0, height - 1)) - center_y
    new_coords_x, new_coords_y = torch.meshgrid(new_coords_x, new_coords_y)
    new_coords = torch.stack([new_coords_x.flatten(), new_coords_y.flatten()], dim=0)
    rotated_coords = torch.mm(rotation_matrix, new_coords).round()

    # Calculate new image size
    new_width = int(rotated_coords[0].max() - rotated_coords[0].min() + 1)
    new_height = int(rotated_coords[1].max() - rotated_coords[1].min() + 1)
    scale_w = new_width / width
    scale_h = new_height / height

    # Generate grid of coordinates
    grid = F.affine_grid(torch.tensor([[torch.cos(angle) * scale_w, -torch.sin(angle) / aspect_ratio * scale_h, 0],
                                       [torch.sin(angle) * aspect_ratio * scale_w, torch.cos(angle) * scale_h, 0]],
                                      device=image.device).unsqueeze(0),
                         torch.Size((1, 1, new_height, new_width), device=image.device), align_corners=False)
    # Perform grid sample using bilinear interpolation
    rotated_image = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid, align_corners=False)
    return rotated_image.squeeze()
