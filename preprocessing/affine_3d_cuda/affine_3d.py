import torch

from . import rotate_image_cpp


def compute_rotation_matrix(angles):
    """
    Compute the rotation matrix R=RzRyRx based on angles.

    Args:
        angles (torch.Tensor, float): Angles in radians for rotation along x, y, and z axes.

    Returns:
        torch.Tensor: Computed rotation matrix.
    """
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

    rotation_matrix = torch.mm(rotation_matrix_z, torch.mm(rotation_matrix_y, rotation_matrix_x))
    return rotation_matrix


def get_image_center(image_size):
    """Calculates

        Args:
            image_size (torch.Size): Input image size (depth, height, width).


        Returns:
            tuple (float): center of image (x-width, y-height, z-depth)
    """
    depth, height, width = image_size
    return width * 0.5, height * 0.5, depth * 0.5


def get_new_image_info(image_size, angles, translations=torch.tensor([0., 0., 0.]), keep_original_size=False):
    """Calculates new size and position of an image after applying specified rotations and translations.

    Args:
        image_size (torch.Size): Input image size (depth, height, width).
        angles (torch.Tensor): Tuple containing rotation angles along x, y, and z axes in radians.
        translations (torch.Tensor, optional): Translation vector along x, y, and z axes. Defaults to torch.tensor([0., 0., 0.]).
        keep_original_size (bool, optional): Whether to keep the original size of the image. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - [new_depth, new_height, new_width]: New size of the image after transformation.
            - [x_min, y_min, z_min]: Minimum coordinates of the transformed image.
    """
    depth, height, width = image_size
    center_x, center_y, center_z = get_image_center(image_size)

    # Rotation matrix
    rotation_matrix = compute_rotation_matrix(angles)

    # Calculate coordinates of vertices after rotation
    coords_x = torch.tensor((0, width - 1)) - center_x
    coords_y = torch.tensor((0, height - 1)) - center_y
    coords_z = torch.tensor((0, depth - 1)) - center_z
    coords_x, coords_y, coords_z = torch.meshgrid(coords_x, coords_y, coords_z)
    coords = torch.stack([coords_x.flatten(), coords_y.flatten(), coords_z.flatten()], dim=0)
    rotated_coords = torch.mm(rotation_matrix, coords)

    # Calculate new image position (xyz), size (WHD)
    new_max = rotated_coords.max(dim=1).values
    new_min = rotated_coords.min(dim=1).values
    new_size = (new_max - new_min + 1).int()

    # Recalculate the new image size and position if keeping the original image size
    if keep_original_size:
        new_min += (new_size - torch.tensor([width, height, depth])) * 0.5
        new_size = torch.tensor([width, height, depth])

    # Apply translations by shifting the new image position
    new_min -= translations
    return new_size.flip(0).numpy().tolist(), new_min.numpy().tolist()


def affine_image_3d_cuda(image, angles=torch.tensor([0., 0., 0.]), translations=torch.tensor([0., 0., 0.]),
                         interpolation_mode="nearest", keep_original_size=False):
    # Prepare orig image
    assert image.device.type == "cuda", "The input image has to be a tensor on cuda."
    if not image.is_contiguous():
        image = image.contiguous()
        torch.cuda.empty_cache()

    # Get the new image size (DHW) and min coordinates (xyz) after rotation (axis, center of image) and translation
    new_size, new_min_coords = get_new_image_info(image.size(), angles, translations, keep_original_size)
    # Container for new image
    output_tensor = torch.zeros(size=new_size, dtype=image.dtype, device=image.device)
    # Interpolation mode
    if interpolation_mode == "trilinear":
        interpolation = rotate_image_cpp.InterpolationMethod.TRILINEAR
    else:
        interpolation = rotate_image_cpp.InterpolationMethod.NEAREST

    # Call rotation function
    rotate_image_cpp.rotate_image(image, output_tensor, *new_min_coords,
                                  *angles.numpy().tolist(), interpolation)
    return output_tensor


def affine_position_3d(image_size, angles, translations, indexes, keep_original_size=False):
    """
    Calculate the corresponding indexes in the rotated image from indexes in the original image.

    Args:
    - image_size (Torch.Size): Size of the original image (depth(z), height(y), width(x))
    - angles (torch.Tensor, float): Rotation angles around the x, y, z-axis in radians, shape (3,), clockwise
    - translations (torch.Tensor, float): Translation distances along the x, y, z-axis in pixels, shape (3,)
    - indexes (torch.Tensor, float): Indexes (x-width, y-height, z-depth) in the original image, shape (N, 3)
    - keep_original_size (bool): if the transformed image has the original image size

    Returns:
    - rotated_indexes (torch.Tensor, float): Corresponding indexes (width, height, depth) in the rotated image, shape (N, 3)
    """

    # Rotation matrix
    rotation_matrix = compute_rotation_matrix(angles)

    # Apply translation and rotation to indexes
    indexes = indexes - torch.tensor(get_image_center(image_size))
    rotated_indexes = torch.mm(rotation_matrix, indexes.float().T)

    # get the new image size (DHW) and min coordinates (xyz) after rotation (axis, center of image) and translation
    _, new_min_coords = get_new_image_info(image_size, angles, translations, keep_original_size)

    return (rotated_indexes - torch.tensor(new_min_coords).view(-1, 1)).round().T
