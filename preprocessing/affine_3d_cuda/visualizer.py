import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np


def vis(img, img_transformed, idx=None, idx_transformed=None, save_path=None):
    if idx is None or idx_transformed is None:
        idx = (i//2 for i in img.shape[::-1])
        idx_transformed = (i//2 for i in img_transformed.shape[::-1])

    if torch.is_tensor(idx):
        idx = idx[0].numpy().tolist()
        idx_transformed = idx_transformed[0].numpy().tolist()

    x, y, z = idx
    xr, yr, zr = idx_transformed

    # x-y
    sl = img[int(z), :, :].astype(np.uint8)  # image d h w
    sl_color = cv2.cvtColor(sl, cv2.COLOR_GRAY2RGB)
    cv2.circle(sl_color, (round(x), round(y)), 5, (0, 255, 0))
    sl_rotated = img_transformed[round(zr), :, :].astype(np.uint8)  # image d h w
    sl_rotated_color = cv2.cvtColor(sl_rotated, cv2.COLOR_GRAY2RGB)
    cv2.circle(sl_rotated_color, (round(xr), round(yr)), 5, (0, 255, 0))
    # x-z
    sl2 = img[:, int(y), :].astype(np.uint8)  # image d h w
    sl2_color = cv2.cvtColor(sl2, cv2.COLOR_GRAY2RGB)
    cv2.circle(sl2_color, (round(x), round(z)), 5, (0, 255, 0))
    sl2_rotated = img_transformed[:, int(yr), :].astype(np.uint8)  # image d h w
    sl2_rotated_color = cv2.cvtColor(sl2_rotated, cv2.COLOR_GRAY2RGB)
    cv2.circle(sl2_rotated_color, (round(xr), round(zr)), 5, (0, 255, 0))
    # y-z
    sl3 = img[:, :, int(x)].astype(np.uint8)  # image d h w
    sl3_color = cv2.cvtColor(sl3, cv2.COLOR_GRAY2RGB)
    cv2.circle(sl3_color, (round(y), round(z)), 5, (0, 255, 0))
    sl3_rotated = img_transformed[:, :, int(xr)].astype(np.uint8)  # image d h w
    sl3_rotated_color = cv2.cvtColor(sl3_rotated, cv2.COLOR_GRAY2RGB)
    cv2.circle(sl3_rotated_color, (round(yr), round(zr)), 5, (0, 255, 0))

    plt.figure(figsize=(20, 15))
    plt.subplot(1, 6, 1)
    plt.imshow(sl_color)  # Show a slice with lm of the original image
    plt.title('Original Image (x-y)')
    plt.axis('off')
    plt.subplot(1, 6, 2)
    plt.imshow(sl_rotated_color)  # Show a slice with lm of the rotated image
    plt.title('Transformed Image (x-y)')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(sl2_color)  # Show a slice with lm of the original image
    plt.title('Original Image (x-z)')
    plt.axis('off')
    plt.subplot(1, 6, 4)
    plt.imshow(sl2_rotated_color)  # Show a slice with lm of the rotated image
    plt.title('Transformed Image (x-z)')
    plt.axis('off')
    plt.subplot(1, 6, 5)
    plt.imshow(sl3_color)  # Show a slice with lm of the original image
    plt.title('Original Image (y-z)')
    plt.axis('off')
    plt.subplot(1, 6, 6)
    plt.imshow(sl3_rotated_color)  # Show a slice with lm of the rotated image
    plt.title('Transformed Image (y-z)')
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

