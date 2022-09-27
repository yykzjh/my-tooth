import torch
from torch import nn as nn

from lib.losses.utils import expand_as_one_hot, compute_per_channel_dice




class BaseMeter():
    """
    最基础的评价度量器
    """
    def __init__(self, classes=1, softmax_normalization=True):
        """
        Args:
            classes: 类别数
            softmax_normalization: 是否softmax
        """
        self.classes = classes
        if not softmax_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def get_dice(self, input, target):
        """
        计算dice相似性系数
        Args:
            input: 神经网络输出的概率特征图
            target: 标签图像

        Returns:

        """
        # 将标签进行one-hot编码
        target = expand_as_one_hot(target.long(), self.classes)

        # 判断特征图和标签图的维度和大小是否一致
        assert input.dim() == target.dim() == 5, "'input' and 'target' have different number of dims"
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # 将概率特征图进行softmax
        input = self.normalization(input)

        # 计算各个类别(通道)的dice相似性系数
        per_channel_dice = compute_per_channel_dice(input, target)

        return per_channel_dice.detach().cpu().numpy()















