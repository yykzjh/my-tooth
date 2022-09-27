"""
3D牙齿数据集、 DenseVNet网络、 adam优化器、 Dice Loss配置参数
"""

"""
************************************************************************************************************************
————————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
CUDA_VISIBLE_DEVICES = "0"  # 选择可用的GPU编号

seed = 1777777  # 随机种子

cuda = True  # 是否使用GPU

benchmark = False  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

deterministic = True  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

"""
************************************************************************************************************************
————————————————————————————————————————————————     预处理       ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
resample_spacing = [0.5, 0.5, 0.5]  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
# [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

clip_lower_bound = 30  # clip的下边界百分位点，图像中每个体素的数值按照从大到小排序，其中小于30%分位点的数值都等于30%分位数
clip_upper_bound = 99.9  # clip的上边界百分位点，图像中每个体素的数值按照从大到小排序，其中大于100%分位点的数值都等于100%分位数

normalization = "full_volume_mean"  # 采用的归一化方法，可选["full_volume_mean","mean","max","max_min"]。其中"full_volume_mean"
# 采用的是整个图像计算出的均值和标准差,"mean"采用的是图像前景计算出的均值和标准差,"max"是将图像所有数值除最大值,"max_min"采用的是Min-Max归一化

samples_train = 2048  # 作为实际的训练集采样的子卷数量，也就是在原训练集上随机裁剪的子图像数量
samples_val = 256  # 作为实际的验证集采样的子卷数量，也就是在原验证集上随机裁剪的子图像数量

crop_size = (160, 160, 96)  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
# 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

crop_threshold = 0.1  # 随机裁剪时需要满足的条件，不满足则重新随机裁剪的位置。条件表示的是裁剪区域中的前景占原图总前景的最小比例

"""
************************************************************************************************************************
————————————————————————————————————————————————     数据增强    ————————————————————————————————————————————————————————
************************************************************************************************************************
"""
augmentation = True  # 训练集是否需要做数据增强，验证集默认不做数据增强

augmentation_probability = 0.3  # 每张图像做数据增强的概率

# 弹性形变参数
elastic_transform_sigma = 20  # 高斯滤波的σ,值越大，弹性形变越平滑
elastic_transform_alpha = 1  # 形变的幅度，值越大，弹性形变越剧烈

# 高斯噪声参数
gaussian_noise_mean = 0  # 高斯噪声分布的均值
gaussian_noise_std = 0.01  # 高斯噪声分布的标准差,值越大噪声越强

# 随机缩放参数
random_rescale_min_percentage = 0.5  # 随机缩放时,最小的缩小比例
random_rescale_max_percentage = 1.5  # 随机缩放时,最大的放大比例

# 随机旋转参数
random_rotate_min_angle = -50  # 随机旋转时,反方向旋转的最大角度
random_rotate_max_angle = 50  # 随机旋转时,正方向旋转的最大角度

# 随机位移参数
random_shift_max_percentage = 0.3  # 在图像的三个维度(D,H,W)都进行随机位移，位移量的范围为(-0.3×(D、H、W),0.3×(D、H、W))

"""
************************************************************************************************************************
————————————————————————————————————————————————     数据读取     ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
dataset_name = "3DTooth"  # 数据集名称， 可选["3DTooth", ]

dataset_path = r"./datasets/src_10"  # 数据集路径

batch_size = 4  # batch_size大小

num_workers = 4  # num_workers大小

"""
************************************************************************************************************************
————————————————————————————————————————————————     网络模型      ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
model_name = "DENSEVNET"  # 模型名称，可选["DENSEVNET","VNET"]

in_channels = 1  # 模型最开始输入的通道数,即模态数

classes = 35  # 模型最后输出的通道数,即类别总数

"""
************************************************************************************************************************
————————————————————————————————————————————————      优化器      ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
optimizer_name = "adam"  # 优化器名称，可选["adam","sgd","rmsprop"]

learning_rate = 0.001  # 学习率

weight_decay = 0.00001  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

momentum = 0.5  # 动量大小

"""
************************************************************************************************************************
————————————————————————————————————————————————     损失函数     ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
loss_function_name = "DiceLoss"  # 损失函数名称，可选["DiceLoss","CrossEntropyLoss","WeightedCrossEntropyLoss","MSELoss",
# "SmoothL1Loss","L1Loss","WeightedSmoothL1Loss","BCEDiceLoss","BCEWithLogitsLoss"]

class_weight = [0.1, 0.3, 3] + [1.0] * (classes - 3)  # 各类别计算损失值的加权权重

sigmoid_normalization = False  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

skip_index_after = None  # 从某个索引的通道(类别)后不计算损失值

"""
************************************************************************************************************************
————————————————————————————————————————————————    训练相关参数   ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
runs_dir = r"./runs"  # 运行时产生的各类文件的存储根目录

start_epoch = 0  # 训练时的起始epoch
end_epoch = 50  # 训练时的结束epoch

best_dice = 0.60  # 保存检查点的初始条件

terminal_show_freq = 50  # 终端打印统计信息的频率,以step为单位

"""
************************************************************************************************************************
————————————————————————————————————————————————    测试相关参数   ———————————————————————————————————————————————————————
************************************************************************************************************************
"""
test_type = 1  # 测试类型,0为无标签的单张图像,1为有标签的单张图像，2为图像数据集

single_image_path = r"./datasets/src_10/train/images/15_2.nrrd"  # 单张图像的存储路径

single_label_path = "./datasets/src_10/train/labels/15_2.nrrd"  # 单张图像对应的标签存储路径

test_dataset_path = r"./datasets/src_10/val"  # 测试数据集的路径

test_batch_size = 1  # 测试数据集读取时的batch_size

test_num_workers = 1  # 测试数据集读取时的num_workers

crop_stride = [4, 4, 4]

















