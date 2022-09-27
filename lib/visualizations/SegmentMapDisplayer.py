# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/27 20:14
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import numpy as np
from collections import Counter

from lib import utils
import configs.config as config

import matplotlib.pyplot as plt
from mayavi import mlab




# 展示预测分割图和标注图的分布直方图对比
def display_compare_hist(label1, label2):
    # 获取类别标签个数
    bins = config.classes
    # 设置解决中文乱码问题
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 将数据扁平化
    label1_flatten = label1.flatten()
    label2_flatten = label2.flatten()

    # 初始化统计数据
    count1 = np.zeros((bins, ))
    count2 = np.zeros((bins, ))

    # 统计标签数值
    label1_count = Counter(label1_flatten)
    label2_count = Counter(label2_flatten)

    # 赋值给统计数据
    for num, cnt in label1_count.items():
        num = int(num)
        if num != 0:
            count1[num] = cnt
    for num, cnt in label2_count.items():
        num = int(num)
        if num != 0:
            count2[num] = cnt

    # 定义柱状的宽度
    width = 0.3

    # 画图
    plt.bar([i - width / 2 for i in range(bins)], count1, width=width, label='Label')
    plt.bar([i + width / 2 for i in range(bins)], count2, width=width, label='Pred')

    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('标签图和预测图柱状分布对比')
    # 设置轴上的标题
    plt.xlabel("类别标签值")
    plt.ylabel("统计数量")
    # 设置x轴注释
    plt.xticks([i for i in range(bins)], list(range(bins)))

    plt.show()





# 分割图的3D可视化
def display_segmentation_3D(class_map):
    # 读取索引文件
    index_to_class_dict = utils.load_json_file(
        os.path.join(r"./lib/dataloaders/index_to_class", config.dataset_name + ".json"))
    class_to_index_dict = {}
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key



    x = []
    y = []
    z = []
    label = []
    for index in [int(index) for index in index_to_class_dict.keys()]:
        # print(index)
        # print(np.where(class_map == i))
        pos_x, pos_y, pos_z = np.nonzero(class_map == index)
        x.extend(list(pos_x))
        y.extend(list(pos_y))
        z.extend(list(pos_z))
        label.extend([index] * len(pos_x))

    color_table = np.array([
        [255, 255, 255, 0],  # 0 background
        [255, 255, 255, 30],  # 1 gum
        [255, 215, 0, 255],  # 2 implant
        [85, 0, 0, 255],  # 3 ul1
        [255, 0, 0, 255],  # 4 ul2
        [85, 85, 0, 255],  # 5 ul3
        [255, 85, 0, 255],  # 6 ul4
        [85, 170, 0, 255],  # 7, ul5
        [255, 170, 0, 255],  # 8, ul6
        [85, 255, 0, 255],  # 9 ul7
        [255, 255, 0, 255],  # 10, ul8
        [0, 0, 255, 255],  # 11 ur1
        [170, 0, 255, 255],  # 12 ur2
        [0, 85, 255, 255],  # 13 ur3
        [170, 85, 255, 255],  # 14 ur4
        [0, 170, 255, 255],  # 15 ur5
        [170, 170, 255, 255],  # 16 ur6
        [0, 255, 255, 255],  # 17 ur7
        [170, 255, 255, 255],  # 18 ur8
        [0, 0, 127, 255],  # 19 bl1
        [170, 0, 127, 255],  # 20 bl2
        [0, 85, 127, 255],  # 21 bl3
        [170, 85, 127, 255],  # 22 bl4
        [0, 170, 127, 255],  # 23 bl5
        [170, 170, 127, 255],  # 24 bl6
        [0, 255, 127, 255],  # 25 bl7
        [170, 255, 127, 255],  # 26 bl8
        [0, 0, 0, 255],  # 27 br1
        [170, 0, 0, 255],  # 28 br2
        [0, 85, 0, 255],  # 29 br3
        [170, 85, 0, 255],  # 30 br4
        [0, 170, 0, 255],  # 31 br5
        [170, 170, 0, 255],  # 32 br6
        [0, 255, 0, 255],  # 33 br7
        [170, 255, 0, 255],  # 34 br8
    ], dtype=np.uint8)
    extent = [0, class_map.shape[0], 0, class_map.shape[1], 0, class_map.shape[2]]

    p3d = mlab.points3d(x, y, z, label, extent=extent, mode='cube', scale_factor=0.5, scale_mode="none")
    mlab.xlabel("X Label")
    mlab.ylabel("Y Label")
    mlab.zlabel("Z Label")
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = config.classes
    p3d.module_manager.scalar_lut_manager.lut.table = color_table
    p3d.glyph.color_mode = "color_by_scalar"
    p3d.glyph.glyph_source.glyph_source.center = [0, 0, 0]
    mlab.show()





