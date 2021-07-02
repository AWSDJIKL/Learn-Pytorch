# -*- coding: utf-8 -*-
'''
OGlt数据集导入
'''
# @Time : 2021/6/6 14:52 
# @Author : LINYANZHEN
# @File : OGlt.py
import pickle
import os
import random

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


class OGltDataset(torch.utils.data.Dataset):
    train = []
    test = []

    @classmethod
    def shuffle_train_test(cls, train_rate=0.7):
        cls.train = []
        cls.test = []
        # 一共1623个不同的字母，每个字母都由20个人绘制，每个字母都抽取不同的人
        for i in range(1623):
            # 在20人中分割出训练集和测试集
            l = list(range(1, 21))
            random.shuffle(l)
            cls.train.append([l.pop() for i in range(int(20 * train_rate))])
            cls.test.append(l)

    def __init__(self, data_dir, mode="train", img_transforms=None, loader=utils.image_loader):
        self.image_path_list = []
        self.label_list = []
        if mode == "train":
            # 记录第几个字母
            i = 0
            for root, dirs, files in os.walk(data_dir):
                if len(files) == 20:
                    for file in files:
                        # 图片命名为 4位数类ID_人编号.png
                        print(file)
                        print(self.train[i])
                        if int(file[-6:-4]) in self.train[i]:
                            self.image_path_list.append(os.path.join(root, file))
                            self.label_list.append(i)
                    i += 1
        elif mode == "test":
            # 记录第几个字母
            i = 0
            for root, dirs, files in os.walk(data_dir):
                if len(files) == 20:
                    for file in files:
                        # 图片命名为 4位数类ID_人编号.png
                        if int(file[-6:-4]) in self.test[i]:
                            self.image_path_list.append(os.path.join(root, file))
                            self.label_list.append(i)
                    i += 1
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
    data_dir = "G:\Dataset\Omniglot"
    OGltDataset.shuffle_train_test(train_rate=0.7)
    train_loader = torch.utils.data.DataLoader(OGltDataset(data_dir, mode="train"))
    val_loader = torch.utils.data.DataLoader(OGltDataset(data_dir, mode="test"))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
