import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from nibabel.viewers import OrthoSlicer3D
from collections import Counter

import configs.config as config

from lib import utils
from lib.visualizations.AverageMeterWriter import AverageMeterWriter


class Tester():
    """
    Tester class
    """
    def __init__(self, model, meter, device):
        """
        Args:
            model: 网络模型
            meter: 评价指标度量器
            device: 设备
        """
        # 传入的参数
        self.model = model
        self.meter = meter
        self.device = device

        # 初始化指标统计打印工具
        self.writer = AverageMeterWriter()


    def split_test(self, image):
        # 获取图像尺寸
        ori_shape = image.size()[2:]
        # 初始化输出的特征图
        output = torch.zeros((image.size()[0], config.classes, *ori_shape), device=self.device)
        # 切片的大小
        slice_shape = config.crop_size
        # 在三个维度上滑动的步长
        stride = config.crop_stride

        # 在三个维度上进行滑动切片
        for shape0_start in range(0, ori_shape[0], stride[0]):
            shape0_end = shape0_start + slice_shape[0]
            start0 = shape0_start
            end0 = shape0_end
            if shape0_end >= ori_shape[0]:
                end0 = ori_shape[0]
                start0 = end0 - slice_shape[0]

            for shape1_start in range(0, ori_shape[1], stride[1]):
                shape1_end = shape1_start + slice_shape[1]
                start1 = shape1_start
                end1 = shape1_end
                if shape1_end >= ori_shape[1]:
                    end1 = ori_shape[1]
                    start1 = end1 - slice_shape[1]

                for shape2_start in range(0, ori_shape[2], stride[2]):
                    shape2_end = shape2_start + slice_shape[2]
                    start2 = shape2_start
                    end2 = shape2_end
                    if shape2_end >= ori_shape[2]:
                        end2 = ori_shape[2]
                        start2 = end2 - slice_shape[2]

                    slice_tensor = image[:, :, start0:end0, start1:end1, start2:end2]
                    slice_predict = self.model(slice_tensor.to(self.device))
                    # slice_predict = nn.Softmax(dim=1)(slice_predict)
                    output[:, :, start0:end0, start1:end1, start2:end2] += slice_predict

                    if shape2_end >= ori_shape[2]:
                        break

                if shape1_end >= ori_shape[1]:
                    break

            if shape0_end >= ori_shape[0]:
                break

        return output



    def test_single_image_without_label(self, image):
        self.model.eval()

        with torch.no_grad():
            # 维度扩张
            image = torch.FloatTensor(image.numpy()).unsqueeze(0).unsqueeze(0)
            # 将图像放到GPU上
            image = image.to(self.device)
            # 输入到网络
            output = self.split_test(image)
            # 对各通道进行softmax
            probability_map = nn.Softmax(dim=1)(output)
            # 根据各通道的概率，选择最大值所在的索引作为该处的类别
            class_map = torch.argmax(probability_map, dim=1).squeeze(0).cpu().numpy()
            # 显示
            OrthoSlicer3D(class_map).show()


    def test_single_image(self, image, label):
        self.writer.reset()
        self.model.eval()

        with torch.no_grad():
            # 维度扩张
            image = torch.FloatTensor(image.numpy()).unsqueeze(0).unsqueeze(0)
            label = torch.FloatTensor(label.numpy()).unsqueeze(0)
            # 将图像放到GPU上
            image = image.to(self.device)
            label = label.to(self.device)
            # 输入到网络
            output = self.split_test(image)
            # 计算各通道的dsc
            per_channel_dice = self.meter.get_dice(output, label)
            # 统计并打印
            self.writer.update(per_channel_dice)
            self.writer.display_terminal()
            # 对各通道进行softmax
            probability_map = nn.Softmax(dim=1)(output)
            # 将训练样本中没有的类别概率置为0
            probability_map[:, [26, 34], ...] = 0
            # 根据各通道的概率，选择最大值所在的索引作为该处的类别
            class_map = torch.argmax(probability_map, dim=1).squeeze(0).cpu().numpy()
            # 转换格式
            label_np = label.squeeze(0).cpu().numpy()

            # # 输出概率图
            # print(output[0, :, class_map == 26].permute(1, 0)[:20, :])
            # print(output[0, :, class_map == 26].permute(1, 0)[:20, 26].reshape((-1, 1)))

            # 显示分布对比图
            self.display_compare_hist(label_np, class_map, bins=35)


            OrthoSlicer3D(label_np).show()
            OrthoSlicer3D(class_map).show()


    def test_image_set(self, test_loader):
        self.writer.reset()
        self.model.eval()

        for image, label in tqdm(test_loader):
            with torch.no_grad():
                # 将图像放到GPU上
                image = image.to(self.device)
                label = label.to(self.device)
                # 输入到网络
                output = self.split_test(image)
                # 计算各通道的dsc
                per_channel_dice = self.meter.get_dice(output, label)
                # 统计并打印
                self.writer.update(per_channel_dice)
        self.writer.display_terminal()



    # 展示预测分割图和标注图的分布直方图对比
    def display_compare_hist(self, label1, label2, bins=35):
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
    def display_segmentation_3D(self, class_map):
        plt.scatter()









