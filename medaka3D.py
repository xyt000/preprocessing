import os

import SimpleITK as sitk
import cv2
import numpy as np
from torchio.transforms import Affine
import pandas as pd


def getLandmarksFromTXTFile(file, split=','):
    """
    Extract each landmark point line by line from a text file, and return
    vector containing all landmarks.
    """
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split('\n')[0].split(split)])
        landmarks = np.asarray(landmarks).reshape((-1, 3))
        return landmarks


def getLandmarksFromExcelFile(file, scale=1.0):
    dat = pd.read_excel(file)
    return {name: [x * scale, y * scale, z * scale] for name, x, y, z in zip(dat.landmark, dat.x, dat.y, dat.z)}


if __name__ == '__main__':
    # affine transform
    rxs = []
    rys = []
    rzs = range(0, 360, 5)
    pad_size = -1
    save_data = True
    save_folder = '/home/ws/ml0077/work/data/medaka/augmentation/data3'
    for rz in rzs:
        rotation_degrees = [0, 0, rz]
        translations = [0, 0, 0]
        rotation_radians = np.radians(rotation_degrees).tolist()
        img_rotation = Affine(scales=[1.0, 1.0, 1.0], degrees=rotation_degrees, translation=translations,
                              center='image')
        pos_rotation = sitk.Euler3DTransform()
        pos_rotation.SetRotation(*rotation_radians)
        pos_rotation.SetTranslation(translations)

        # data
        vol_name = 'example_landmarks_803_4-2_scale2_crop' # 'test_reference_1263_94-1_scale2_crop' #
        img_path = f'/home/ws/ml0077/work/data/medaka/{vol_name}.tif'
        landmark_path = f'/home/ws/ml0077/work/data/medaka/{vol_name}.xlsx'
        landmark_names = ['mandible dentry', 'hyoid fusion', 'first vertebra', 'optic nerve head R',
                          'optic nerve head L']
        sitk_image = sitk.ReadImage(img_path, sitk.sitkFloat32)
        # np_image = sitk.GetArrayFromImage(sitk_image)
        # # threshold image between p10 and p98 then re-scale [0-255]
        # p0 = np_image.min().astype('float')
        # p10 = np.percentile(np_image, 10)
        # p99 = np.percentile(np_image, 99)
        # p100 = np_image.max().astype('float')
        # sitk_image = sitk.Threshold(sitk_image,
        #                             lower=p10,
        #                             upper=p100,
        #                             outsideValue=p10)
        # sitk_image = sitk.Threshold(sitk_image,
        #                             lower=p0,
        #                             upper=p99,
        #                             outsideValue=p99)
        # sitk_image = sitk.RescaleIntensity(sitk_image,
        #                                    outputMinimum=0,
        #                                    outputMaximum=255)
        from skimage import io

        # Convert from [depth, width, height] to [height/row, width/col, depth]
        image_data = sitk.GetArrayFromImage(sitk_image).transpose(2, 1, 0).astype('uint8')
        # padding
        image_dims = np.shape(image_data)
        if pad_size > 0:
            image_padded = np.zeros((image_dims[0] + 2*pad_size, image_dims[1] + 2*pad_size, image_dims[2]), dtype=np.uint8)
            image_padded[pad_size:image_dims[0] + pad_size, pad_size:image_dims[1] + pad_size, :] = image_data
            image_dims = np.shape(image_padded)
            image_data = image_padded
        # rotate image
        image_rotated = img_rotation(np.expand_dims(image_data, axis=0))[0]
        if save_data:
            rx, ry, rz = rotation_degrees
            tx, ty, tz = translations
            result_image = sitk.GetImageFromArray(image_rotated.transpose(2, 1, 0))
            #result_image .CopyInformation(sitk_image)
            sitk.WriteImage(result_image, os.path.join(save_folder, f'rx{rx}_ry{ry}_rz{rz}_tx{tx}_ty{ty}_tz{tz}.tif'))
        # set the rotate center to the middle of the image

        center = [(image_dims[0] - 1) / 2, (image_dims[1] - 1) / 2 - 1, (image_dims[2] - 1) / 2]
        pos_rotation.SetCenter(center)

        # all_landmarks = getLandmarksFromTXTFile(landmark_path, split=",")
        all_landmarks = getLandmarksFromExcelFile(landmark_path, scale=0.5)
        all_landmarks_rotated = {}
        for lm_name in landmark_names:
            if pad_size > 0:
                lm = [p + pad for p, pad in zip(all_landmarks[lm_name], [pad_size, pad_size, 0])]
            else:
                lm = all_landmarks[lm_name]

            lm_rotated = pos_rotation.TransformPoint(lm)

            # visualization
            #
            sl = image_data[:, :, round(lm[2])].astype(np.uint8)
            sl_color = cv2.cvtColor(sl, cv2.COLOR_GRAY2RGB)
            cv2.circle(sl_color, (round(lm[1]), round(lm[0])), 2, (0, 255, 0))

            sl_rotated = image_rotated[:, :, round(lm_rotated[2])].astype(np.uint8)
            sl_rotated_color = cv2.cvtColor(sl_rotated, cv2.COLOR_GRAY2RGB)
            cv2.circle(sl_rotated_color, (round(lm_rotated[1]), round(lm_rotated[0])), 2, (0, 255, 0))

            cv2.imshow(f'{lm_name}', sl_color)
            cv2.imwrite(f'{lm_name}.png', sl_color)
            cv2.imshow(f'rotated {rz}', sl_rotated_color)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            all_landmarks_rotated.update({lm_name: (lm_rotated[0], lm_rotated[1], lm_rotated[2])})
        if save_data:
            pd.DataFrame(all_landmarks_rotated).to_csv(os.path.join(save_folder, f'rx{rx}_ry{ry}_rz{rz}_tx{tx}_ty{ty}_tz{tz}.csv'))
