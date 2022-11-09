import os
import scipy
import nrrd
import glob
import torch
import nibabel as nib
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    # 对两个统计字典按键排序
    class_statistic_dict = dict(sorted(class_statistic_dict.items(), key=lambda x: x[0]))
    label_statistic_dict = dict(sorted(label_statistic_dict.items(), key=lambda x: x[0]))

    # 设置解决中文乱码问题
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    # 定义直方图条形宽度
    width = 0.8



    # 获得每类牙齿的统计数量
    tooth_num = [sub_dict["tooth_num"] for class_index, sub_dict in class_statistic_dict.items() if class_index != 0]
    # 展示各类牙齿的数量对比直方图
    plt.bar([i for i in range(1, config.classes)],
            tooth_num,
            width=width,
            tick_label=[index_to_class_dict[i] for i in range(1, config.classes)],
            label='牙齿数量')
    # 设置y轴显示范围
    plt.ylim([0, max(tooth_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(tooth_num):
        plt.text(i + 1, num, num, ha='center', fontsize=12)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('不同种类的牙齿数量对比')
    # 设置轴上的标题
    plt.xlabel("牙齿类别")
    plt.ylabel("数量(颗)")
    plt.show()



    # 获得每类牙齿的体素的统计数量
    voxel_num = [sub_dict["voxel_num"] for class_index, sub_dict in class_statistic_dict.items() if class_index != 0]
    # 展示各类牙齿的体素数量对比直方图
    plt.bar([i for i in range(1, config.classes)],
            voxel_num,
            width=width,
            tick_label=[index_to_class_dict[i] for i in range(1, config.classes)],
            label='牙齿体素数量')
    # 设置y轴显示范围
    plt.ylim([0, max(voxel_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(voxel_num):
        plt.text(i + 1, num, num, ha='center', fontsize=12)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('不同种类的牙齿体素数量对比')
    # 设置轴上的标题
    plt.xlabel("牙齿类别")
    plt.ylabel("体素数量")
    plt.show()


    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16


    # 获得每张图像的牙齿统计数量
    tooth_cnt = [sub_dict["tooth_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获得每张图像是否存在种植体
    exist_implant = [sub_dict["have_implant"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获得每个条形设置的颜色
    bar_colors = ["r" if exist else "b" for exist in exist_implant]
    # 获取文件名称列表
    file_names = [file_name for file_name, sub_dict in label_statistic_dict.items()]
    # 展示每张图像牙齿数量对比直方图
    plt.bar([i for i in range(len(labels))],
            tooth_cnt,
            width=width,
            tick_label=file_names,
            color=bar_colors,
            label='牙齿数量')
    # 设置y轴显示范围
    plt.ylim([0, max(tooth_cnt) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(tooth_cnt):
        plt.text(i, num, num, ha='center', fontsize=16)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('每张图像的牙齿数量对比')
    # 设置轴上的标题
    plt.xlabel("文件名称")
    plt.ylabel("数量(颗)")
    plt.show()



    # 重新设置宽度
    width = 0.4
    # 获得每张图像的前景体素统计数量
    foreground_voxel_num = [sub_dict["foreground_voxel_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获得每张图像的总体素统计数量
    total_voxel_num = [sub_dict["total_voxel_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获取文件名称列表
    file_names = [file_name for file_name, sub_dict in label_statistic_dict.items()]
    # 展示每张图像前景体素对比直方图
    plt.bar([i - width / 2 for i in range(len(labels))],
            foreground_voxel_num,
            width=width,
            tick_label=file_names,
            color="r",
            label='前景体素数量')
    # 展示每张图像总体素对比直方图
    plt.bar([i + width / 2 for i in range(len(labels))],
            total_voxel_num,
            width=width,
            tick_label=file_names,
            color="b",
            label='总体素数量')
    # 设置y轴显示范围
    plt.ylim([0, max(total_voxel_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(foreground_voxel_num):
        plt.text(i - width / 2, num, num, ha='center', fontsize=16)
    for i, num in enumerate(total_voxel_num):
        plt.text(i + width / 2, num, num, ha='center', fontsize=16)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('每张图像的前景体素和总体素数量对比')
    # 设置轴上的标题
    plt.xlabel("文件名称")
    plt.ylabel("体素数量")
    plt.show()



    # 重新设置宽度
    width = 0.8
    # 获得每张图像的slice统计数量
    slice_num = [sub_dict["slice_num"] for file_name, sub_dict in label_statistic_dict.items()]
    # 获取文件名称列表
    file_names = [file_name for file_name, sub_dict in label_statistic_dict.items()]
    # 展示每张图像前景体素对比直方图
    plt.bar([i for i in range(len(labels))],
            slice_num,
            width=width,
            tick_label=file_names,
            label='切片slice数量')
    # 设置y轴显示范围
    plt.ylim([0, max(slice_num) * 1.2])
    # 为每个条形图添加数值标签
    for i, num in enumerate(slice_num):
        plt.text(i, num, num, ha='center', fontsize=16)
    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('每张图像的slice切片数量对比')
    # 设置轴上的标题
    plt.xlabel("文件名称")
    plt.ylabel("切片数量")
    plt.show()


    print(class_statistic_dict)
    print(label_statistic_dict)


















if __name__ == '__main__':
    # show_space_property(r"./datasets/src_10")

    analyse_dataset(dataset_path=r"./datasets/src_10")














