import os
import scipy
import nrrd
import glob
import torch
import nibabel as nib
import seaborn as sns

from lib import dataloaders
from lib import utils
from configs import config




def show_space_property(dataset_path=r"./datasets/src_10"):
    train_images = glob.glob(os.path.join(dataset_path, 'train', "images", '*.nrrd'))
    train_labels = glob.glob(os.path.join(dataset_path, 'train', "labels", '*.nrrd'))
    val_images = glob.glob(os.path.join(dataset_path, 'val', "images", '*.nrrd'))
    val_labels = glob.glob(os.path.join(dataset_path, 'val', "labels", '*.nrrd'))
    keys = ["type", "dimension", "space", "sizes", "space directions", "space origin"]
    values = []
    for i, key in enumerate(keys):
        values.append([])
        for path in train_images:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in train_labels:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in val_images:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        for path in val_labels:
            # print(path)
            _, options = nrrd.read(path)
            values[i].append(options[key])
        print(key + ": ", values[i])



def analyse_dataset(dataset_path=r"./datasets/src_10"):
    # 加载数据集中所有标注图像
    labels_path = glob.glob(os.path.join(dataset_path, "*", "labels", '*.nrrd'))
    # 对所有标注图像进行重采样，统一spacing
    labels = [dataloaders.load_image_or_label(label_path, type="label", viz3d=False) for label_path in labels_path]

    # 加载类别名称和类别索引映射字典
    # 读取索引文件
    index_to_class_dict = utils.load_json_file(
        os.path.join(r"./lib/dataloaders/index_to_class", config.dataset_name + ".json"))
    class_to_index_dict = {}
    # 获得类别到索引的映射字典
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key

    # 初始化类别统计字典
    class_statistic_dict = {class_index: {
        "tooth_num": 0,
        "voxel_num": 0
    } for class_index in range(config.classes)}
    # 初始化每张图像的统计字典
    label_statistic_dict = {os.path.splitext(os.path.basename(label_path))[0]: {
        "tooth_num": 0,
        "have_implant": False,
        "total_voxel_num": 0,
        "foreground_voxel_num": 0,
        "slice_num": 0
    } for label_path in labels_path}

    # 遍历所有标注图像
    for label_i, label in enumerate(labels):
        # 获得当前标注图像的名称
        label_name = os.path.splitext(os.path.basename(labels_path[label_i]))[0]
        # 获取当前标注图像的统计数据
        class_indexes, indexes_cnt = torch.unique(label, return_counts=True)
        # 遍历类别索引，将统计数据累加到类别统计字典
        for i, _ in enumerate(class_indexes):
            class_index = class_indexes[i].item()
            class_statistic_dict[class_index]["tooth_num"] += 1  # 类别计数加1
            class_statistic_dict[class_index]["voxel_num"] += indexes_cnt[i].item()  # 类别体素数累加
        # 更新每张标注图像的统计字典
        label_statistic_dict[label_name]["tooth_num"] = torch.nonzero(class_indexes > 2).shape[0]
        label_statistic_dict[label_name]["have_implant"] = True if 2 in class_indexes else False
        label_statistic_dict[label_name]["total_voxel_num"] = label.numel()
        label_statistic_dict[label_name]["foreground_voxel_num"] = torch.nonzero(label != 0).shape[0]
        label_statistic_dict[label_name]["slice_num"] = label.shape[2]

    print(class_statistic_dict)
    print(label_statistic_dict)


















if __name__ == '__main__':
    # show_space_property(r"./datasets/src_10")

    analyse_dataset(dataset_path=r"./datasets/src_10")














