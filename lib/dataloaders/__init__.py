from torch.utils.data import DataLoader

import configs.config as config

from .preprocess import load_image_or_label
from lib.dataloaders.tooth_dataset import ToothDataset, ToothTestDataset


def get_loader(args):

    # 判断是否加载已有子卷
    load = (not args.createData) or (args.resume is not None)

    # 选择数据集进行加载
    if config.dataset_name == "3DTooth":
        train_set = ToothDataset(mode="train", load=load)
        val_set = ToothDataset(mode="val", load=load)

    else:
        raise "没有这个数据集"


    # 定义数据加载器参数
    params = {
        'train': {
            'batch_size': config.batch_size,
            'shuffle': True,
            'num_workers': config.num_workers
        },
        'val': {
            'batch_size': config.batch_size,
            'shuffle': False,
            'num_workers': config.num_workers
        }
    }

    # 初始化数据加载器
    train_loader = DataLoader(train_set, **params['train'])
    val_loader = DataLoader(val_set, **params['val'])


    return train_loader, val_loader, val_set.full_volume




def get_test_loader():
    # 选择数据集进行加载
    if config.dataset_name == "3DTooth":
       test_set = ToothTestDataset()

    else:
        raise "没有这个数据集"

    # 定义数据加载器参数
    params = {
        'test': {
            'batch_size': config.test_batch_size,
            'shuffle': False,
            'num_workers': config.test_num_workers
        },
    }

    # 初始化数据加载器
    test_loader = DataLoader(test_set, **params['test'])

    return test_loader
























