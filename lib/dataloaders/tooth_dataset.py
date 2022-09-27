import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import lib.augments as augments
import lib.utils as utils
from lib.dataloaders.preprocess import *
from lib.dataloaders.utils import get_viz_set, create_sub_volumes


class ToothDataset(Dataset):
    """
    读取nrrd牙齿数据集
    """

    def __init__(self, mode, load=False):
        """
        Args:
            mode: train/val
            load: 是否加载已生成的子卷数据集
        """
        self.mode = mode
        self.root = config.dataset_path
        self.train_path = os.path.join(self.root, "train")
        self.val_path = os.path.join(self.root, "val")
        self.CLASSES = config.classes
        self.threshold = config.crop_threshold
        self.normalization = config.normalization
        self.augmentation = config.augmentation
        self.list = []
        self.samples = config.samples_train if self.mode == "train" else config.samples_val
        self.full_volume = None

        # 定义数据增强
        if self.augmentation:
            self.transform = augments.RandomChoice(
                transforms=[
                    augments.ElasticTransform(alpha=config.elastic_transform_alpha, sigma=config.elastic_transform_sigma),
                    augments.GaussianNoise(mean=config.gaussian_noise_mean, std=config.gaussian_noise_std),
                    augments.RandomFlip(),
                    augments.RandomRescale(min_percentage=config.random_rescale_min_percentage, max_percentage=config.random_rescale_max_percentage),
                    augments.RandomRotation(min_angle=config.random_rotate_min_angle, max_angle=config.random_rotate_max_angle),
                    augments.RandomShift(max_percentage=config.random_shift_max_percentage)
                ],
                p=config.augmentation_probability
            )

        # 定义子卷根目录
        sub_volume_root_path = os.path.join(
            self.root,
            "sub_volumes",
            self.mode +
            '-vol_' + str(config.crop_size[0]) + 'x' + str(config.crop_size[1]) + 'x' + str(config.crop_size[2]) +
            "-samples_" + str(self.samples)
        )
        # 定义子卷图像保存地址
        self.sub_vol_path = os.path.join(sub_volume_root_path, "generated")
        # 定义子卷图像路径保存的txt地址
        self.list_txt_path = os.path.join(sub_volume_root_path, "list.txt")

        # 直接加载之前生成的数据
        if load:
            self.list = utils.load_list(self.list_txt_path)
            return

        # 创建子卷根目录
        utils.make_dirs(sub_volume_root_path)
        # 创建子卷图像保存地址
        utils.make_dirs(self.sub_vol_path)

        # 分类创建子卷数据集
        if self.mode == 'train':
            images_path_list = sorted(glob.glob(os.path.join(self.train_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.train_path, "labels", "*.nrrd")))

            self.list = create_sub_volumes(images_path_list, labels_path_list, samples=self.samples, sub_vol_path=self.sub_vol_path)

        elif self.mode == 'val':
            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nrrd")))

            self.list = create_sub_volumes(images_path_list, labels_path_list, samples=self.samples, sub_vol_path=self.sub_vol_path)

            self.full_volume = get_viz_set(images_path_list, labels_path_list, image_index=0)

        elif self.mode == 'viz':
            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nrrd")))

            self.full_volume = get_viz_set(images_path_list, labels_path_list, image_index=0)
            self.list = []

        # 保存所有子卷图像路径到txt文件
        utils.save_list(self.list_txt_path, self.list)


    def __len__(self):
        return len(self.list)


    def __getitem__(self, index):
        image_path, label_path = self.list[index]
        image, label = np.load(image_path), np.load(label_path)

        if self.mode == 'train' and self.augmentation:
            augmented_image, augmented_label = self.transform(image, label)

            return torch.FloatTensor(augmented_image.copy()).unsqueeze(0), torch.FloatTensor(augmented_label.copy())

        return torch.FloatTensor(image).unsqueeze(0), torch.FloatTensor(label)






class ToothTestDataset(Dataset):
    """
    测试nrrd牙齿数据集
    """

    def __init__(self):
        self.test_path = config.test_dataset_path

        images_path_list = sorted(glob.glob(os.path.join(self.test_path, "images", "*.nrrd")))
        labels_path_list = sorted(glob.glob(os.path.join(self.test_path, "labels", "*.nrrd")))

        self.list = list(zip(images_path_list, labels_path_list))


    def __getitem__(self, index):
        image_path, label_path = self.list[index]

        image_tensor = load_image_or_label(image_path, type="image", viz3d=False)
        label_tensor = load_image_or_label(label_path, type="label", viz3d=False)

        return torch.FloatTensor(image_tensor.numpy()).unsqueeze(0), torch.FloatTensor(label_tensor.numpy())


    def __len__(self):
        return len(self.list)








