import os
import torch
import argparse
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import configs.config as config

from lib import utils
from lib import dataloaders
from lib import models
from lib import trainers
from lib import losses





def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--createData', action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--pretrain', default=None, type=str, metavar='PATH',
                        help='path to latest pretrain model (default: None)')

    args = parser.parse_args()

    return args



def main():
    print("-----------------------------------------------解析参数--------------------------------------------------")
    args = get_arguments()

    print("------------------------------------设置可用GPU、随机种子、卷积算法优化---------------------------------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    utils.reproducibility()

    print("--------------------------------------------初始化数据加载器-----------------------------------------------")
    train_loader, val_loader, full_volume = dataloaders.get_loader(args)

    print("-------------------------------------------初始化模型和优化器----------------------------------------------")
    model, optimizer = models.create_model(args)

    if config.cuda:
        print("--------------------------------------------将模型放到GPU上-----------------------------------------------")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    print("---------------------------------------------定义损失函数-------------------------------------------------")
    criterion = losses.create_loss(device=device)

    print("---------------------------------------------初始化训练器-------------------------------------------------")
    trainer = trainers.Trainer(model, criterion, optimizer, train_data_loader=train_loader,
                               valid_data_loader=val_loader, lr_scheduler=None, device=device)

    print("----------------------------------------------开始训练---------------------------------------------------")
    trainer.training()




if __name__ == '__main__':
    main()















    # for batch_image, batch_label in train_loader:
    #     image_np = batch_image.squeeze().numpy()
    #     label_np = batch_label.squeeze().numpy()
    #
    #     OrthoSlicer3D(image_np).show()
    #
    #     plt.hist(label_np.flatten(), bins=50, range=(1, 50), color="g", histtype="bar", rwidth=1, alpha=0.6)
    #     plt.show()
    #
    #     print(label_np.shape)
    #
    #     ct = Counter(label_np.flatten())
    #     cc = sorted(ct.items(), key=lambda x: x[0])
    #     cnt = 0
    #     for t in cc:
    #         cnt += 1
    #         if cnt % 5 == 0:
    #             print(t)
    #         else:
    #             print(t, end=", ")
    #     OrthoSlicer3D(label_np).show()








