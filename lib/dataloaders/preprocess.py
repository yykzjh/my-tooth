import numpy as np
import torch
import nrrd
import os
import json
import re
import scipy
from PIL import Image
from scipy import ndimage
from collections import Counter
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt

import configs.config as config

import lib.utils as utils




def load_image_or_label(
        path,
        type=None,
        viz3d=False):
    """
    加载完整标注图像、随机裁剪后的牙齿图像或标注图像
    Args:
        path:路径
        type:标签图或原图
        viz3d:是否用于可视化3D

    Returns:

    """
    # 判断是读取标注文件还是原图像文件
    if type == "label":
        img_np, spacing = load_label(path)
    else:
        img_np, spacing = load_image(path)

    # 重采样
    if config.resample_spacing is not None:
        order = 0 if type == "label" else 3
        img_np = resample_image_spacing(img_np, spacing, config.resample_spacing, order)

    # 直接返回tensor
    if viz3d:
        return torch.from_numpy(img_np)

    # 数值上下界clip
    if type != "label":
        img_np = percentile_clip(img_np, min_val=config.clip_lower_bound, max_val=config.clip_upper_bound)

    # 转换成tensor
    img_tensor = torch.from_numpy(img_np)

    # 归一化
    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()
        img_tensor = normalize_intensity(img_tensor, normalization=config.normalization, norm_values=(MEAN, STD, MAX, MIN))

    return img_tensor





def load_label(path):
    # print(path)
    """
    读取label文件
    Args:
        path: 文件路径

    Returns:

    """
    # 读入 nrrd 文件
    data, options = nrrd.read(path)
    assert data.ndim == 3, "label图像维度出错"

    # 初始化标记字典
    # 读取索引文件
    index_to_class_dict = utils.load_json_file(
        os.path.join(r"./lib/dataloaders/index_to_class", config.dataset_name + ".json"))
    class_to_index_dict = {}
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key
    segment_dict = class_to_index_dict.copy()
    for key in segment_dict.keys():
        segment_dict[key] = {"index": int(segment_dict[key]), "color": None, "labelValue": None}

    for key, val in options.items():
        searchObj = re.search(r'^Segment(\d+)_Name$', key)
        if searchObj is not None:
            segment_id = searchObj.group(1)
            # 获取颜色
            segment_color_key = "Segment" + str(segment_id) + "_Color"
            color = options.get(segment_color_key, None)
            if color is not None:
                tmpColor = color.split()
                color = [int(255 * float(c)) for c in tmpColor]
            segment_dict[val]["color"] = color
            # 获取标签值
            segment_label_value_key = "Segment" + str(segment_id) + "_LabelValue"
            labelValue = options.get(segment_label_value_key, None)
            if labelValue is not None:
                labelValue = int(labelValue)
            segment_dict[val]["labelValue"] = labelValue
    # 替换标签值
    for key, val in segment_dict.items():
        if val["labelValue"] is not None:
            # print(key, val["labelValue"])
            data[data == val["labelValue"]] = -val["index"]
    data = -data

    # # 可视化处理结果
    # data[data == 1] = 40
    #
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
    #
    # OrthoSlicer3D(data).show()

    # 获取体素间距
    spacing = [v[i] for i, v in enumerate(options["space directions"])]

    return data, spacing



def load_image(path):
    """
    加载图像数据
    Args:
        path:路径

    Returns:

    """
    # 读取
    data, options = nrrd.read(path)
    assert data.ndim == 3, "图像维度出错"
    # 修改数据类型
    data = data.astype(np.float64)
    # 获取体素间距
    spacing = [v[i] for i, v in enumerate(options["space directions"])]

    return data, spacing




def resample_image_spacing(data, old_spacing, new_spacing, order):
    """
    根据体素间距对图像进行重采样
    Args:
        data:图像数据
        old_spacing:原体素间距
        new_spacing:新体素间距

    Returns:

    """
    scale_list = [old / new_spacing[i] for i, old in enumerate(old_spacing)]
    return scipy.ndimage.interpolation.zoom(data, scale_list, order=order)



def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy




def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val

    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor



def crop_img(img_tensor, crop_size, crop_point):
    if crop_size[0] == 0:
        return img_tensor
    slices_crop, w_crop, h_crop = crop_point
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_tensor.dim()
    assert inp_img_dim >= 3
    if img_tensor.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img_tensor.shape
    elif img_tensor.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img_tensor.shape
        img_tensor = img_tensor[0, ...]

    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                     h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_tensor.unsqueeze(0)

    return img_tensor




if __name__ == '__main__':
    load_label(r"./datasets/src_10/train/labels/12_2.nrrd")









