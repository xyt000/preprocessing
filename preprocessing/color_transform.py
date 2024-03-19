import torch


def brightness_transform_inplace(image: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    """
    Apply brightness transformation in place to a float torch tensor representing an image.

    Args:
        image (torch.Tensor): Float tensor representing an image.
        brightness_factor (float): Factor to adjust brightness.

    Returns:
        torch.Tensor: Transformed image tensor.

    Raises:
        TypeError: If the input `image` is not a float tensor.

    """
    assert isinstance(image, torch.Tensor) and image.dtype == torch.float32, "Input 'image' must be a float tensor"
    image.mul_(brightness_factor).clamp_(0, 255)


def contrast_transform_inplace(image: torch.Tensor, contrast_factor: float, image_mean=None) -> torch.Tensor:
    """
    Apply contrast transformation to a float torch tensor representing an image.

    Args:
        image (torch.Tensor): Float tensor representing an image.
        contrast_factor (float): Factor to adjust contrast.
        image_mean (float, optional): Mean value of the image. If None, it will be calculated from the image.

    Returns:
        torch.Tensor: Transformed image tensor.

    Raises:
        TypeError: If the input `image` is not a float tensor.

    """
    assert isinstance(image, torch.Tensor) and image.dtype == torch.float32, "Input 'image' must be a float tensor"

    if image_mean is None:
        image_mean = torch.mean(image)

    # Apply contrast transformation in place
    image.sub_(image_mean).mul_(contrast_factor).add_(image_mean).clamp_(0, 255)
