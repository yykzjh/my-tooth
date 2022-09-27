import numpy as np
import torch
import nrrd
import os
import json
import re
from PIL import Image
from scipy import ndimage
from collections import Counter
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt

import configs.config as config

from lib.dataloaders.preprocess import load_image_or_label, crop_img




def create_sub_volumes(*ls, samples, sub_vol_path):
    """
    将数据集中的图像裁剪成子卷，组成新的数据集
    Args:
        *ls: 原图像路径、标注图像路径
        samples: 子卷数量
        sub_vol_path: 子卷存储路径

    Returns:

    """
    # 获取图像数量
    image_num = len(ls[0])
    assert image_num != 0, "Problem reading data. Check the data paths."
    # 先把完整标注图像、完整原图图像读取到内存
    image_tensor_list = []
    label_tensor_list = []
    for image_path in ls[0]:
        # 加载并预处理原图图像
        image_tensor = load_image_or_label(image_path, type="image", viz3d=False)
        # print(image_path)
        # OrthoSlicer3D(image_tensor).show()
        image_tensor_list.append(image_tensor)
    for label_path in ls[1]:
        # 加载并预处理标注图像
        label_tensor = load_image_or_label(label_path, type="label", viz3d=False)
        # OrthoSlicer3D(label_tensor).show()
        label_tensor_list.append(label_tensor)

    # 采样指定数量的子卷
    subvolume_list = []
    for i in range(samples):
        print("id:", i)
        # 随机对某个图像裁剪子卷
        random_index = np.random.randint(image_num)
        # 获取当前图像数据和标签数据
        image_tensor = image_tensor_list[random_index].clone()
        label_tensor = label_tensor_list[random_index].clone()

        # 反复随机生成裁剪区域，直到满足裁剪指标为止
        cnt_loop = 0
        while True:
            cnt_loop += 1
            # 计算裁剪的位置
            crop_point = find_random_crop_dim(label_tensor.shape, config.crop_size)
            # 判断当前裁剪区域满不满足条件
            if find_non_zero_labels_mask(label_tensor, config.crop_threshold, config.crop_size, crop_point):
                # 裁剪
                crop_image_tensor = crop_img(image_tensor, config.crop_size, crop_point)
                # OrthoSlicer3D(crop_image_tensor).show()
                crop_label_tensor = crop_img(label_tensor, config.crop_size, crop_point)
                # data = crop_label_tensor.numpy()
                # plt.hist(data.flatten(), bins=50, range=(1, 50), color="g", histtype="bar", rwidth=1, alpha=0.6)
                # plt.show()
                #
                # print(data.shape)
                #
                # ct = Counter(data.flatten())
                # cc = sorted(ct.items(), key=lambda x: x[0])
                # cnt = 0
                # for t in cc:
                #     cnt += 1
                #     if cnt % 5 == 0:
                #         print(t)
                #     else:
                #         print(t, end=", ")
                # OrthoSlicer3D(data).show()
                print("loop cnt:", cnt_loop, '\n')
                break

        # 存储子卷
        filename = os.path.join(sub_vol_path, 'id_' + str(i) + '-src_id_' + str(random_index) + '-modality')
        image_filename = filename + ".npy"
        np.save(image_filename, crop_image_tensor)
        label_filename = filename + "_seg.npy"
        np.save(label_filename, crop_label_tensor)
        subvolume_list.append((image_filename, label_filename))

    return subvolume_list




def get_viz_set(*ls, image_index=0):
    """

    """
    total_volumes = [
        load_image_or_label(ls[1][image_index], type="image", viz3d=True),
        load_image_or_label(ls[0][image_index], type="label", viz3d=True)
    ]

    return torch.stack(total_volumes, dim=0)




def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)





def find_non_zero_labels_mask(label_map, th_percent, crop_size, crop_point):
    segmentation_map = label_map.clone()
    d1, d2, d3 = segmentation_map.shape
    segmentation_map[segmentation_map > 0] = 1
    total_voxel_labels = segmentation_map.sum()

    cropped_segm_map = crop_img(segmentation_map, crop_size, crop_point)
    crop_voxel_labels = cropped_segm_map.sum()

    label_percentage = crop_voxel_labels / total_voxel_labels
    # print(label_percentage,total_voxel_labels,crop_voxel_labels)
    if label_percentage >= th_percent:
        return True
    else:
        return False












