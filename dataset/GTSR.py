# -*- coding: utf-8 -*-
'''
GTSR数据集导入
'''
# @Time : 2021/6/6 11:21 
# @Author : LINYANZHEN
# @File : GTSR.py
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


class GTSRDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path=None, mode="train", img_transforms=None, loader=utils.image_loader):
        self.image_path_list = []
        self.label_list = []
        if mode == "train":
            for root, dirs, files in os.walk(data_dir):
                for i in files:
                    if i[-3:] == "ppm":
                        self.image_path_list.append(os.path.join(root, i))
                        self.label_list.append(int(i[:5]))
        elif mode == "test":
            # 只读取csv文件即可，将文件名和对应分类提取出来
            data = pd.read_csv(csv_path, sep=";")
            filename = data["Filename"].to_list()
            for i in filename:
                self.image_path_list.append(os.path.join(data_dir, i))
            self.label_list = data["ClassId"].to_list()
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
    train_file_path = "G:\\Dataset\\GTSRB\\GTSRB\\Final_Training\\Images"
    test_file_path = "G:\\Dataset\\GTSRB\\GTSRB\\Final_Test\\Images"
    csv_file_path = "G:\\Dataset\\GTSRB\\GTSRB\\Final_Test\\GT-final_test.csv"

    train_loader = torch.utils.data.DataLoader(GTSRDataset(train_file_path))
    val_loader = torch.utils.data.DataLoader(GTSRDataset(test_file_path, csv_path=csv_file_path, mode="test"))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
