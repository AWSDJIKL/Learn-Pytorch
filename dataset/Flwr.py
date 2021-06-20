# -*- coding: utf-8 -*-
'''
Flwr数据集导入
'''
# @Time : 2021/6/6 12:26 
# @Author : LINYANZHEN
# @File : Flwr.py
import pickle
import os
from PIL import Image
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
import utils
import time
import scipy.io as scio


class FlwrDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode="train", img_transforms=None, loader=utils.image_loader):
        self.image_path_list = []
        self.label_list = []

        if mode == "train":
            for root, dirs, files in os.walk(data_dir):
                for i in files:
                    self.image_path_list.append(os.path.join(root, i))
                    self.label_list.append(int(os.path.split(root)[-1]) - 1)
        elif mode == "val":
            for root, dirs, files in os.walk(data_dir):
                for i in files:
                    self.image_path_list.append(os.path.join(root, i))
                    self.label_list.append(int(os.path.split(root)[-1]) - 1)
        elif mode == "test":
            pass
        # 数据增强
        if img_transforms:
            self.img_transforms = img_transforms
        else:
            # 使用默认的预处理
            self.img_transforms = transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                # transforms.Normalize(means, stds),
            ])
        self.loader = loader
        # 分类类别数，用于最后的全连接分类
        self.num_classes = len(set(self.label_list))

    def __getitem__(self, index):
        path = self.image_path_list[index]
        label = torch.tensor(self.label_list[index], dtype=torch.long)
        img = self.loader(path)
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        return img, label

    def __len__(self):
        return len(self.image_path_list)


def prepare_dataloader():
    train_file_path = "G:\\Dataset\\oxford-102-flower-pytorch\\flower_data\\train"
    val_file_path = "G:\\Dataset\\oxford-102-flower-pytorch\\flower_data\\valid"
    test_file_path = "G:\\Dataset\\oxford-102-flower-pytorch\\flower_data\\test"

    train_loader = torch.utils.data.DataLoader(FlwrDataset(train_file_path, mode="train"))
    val_loader = torch.utils.data.DataLoader(FlwrDataset(val_file_path, mode="val"))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
