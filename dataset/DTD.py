# -*- coding: utf-8 -*-
'''
DTD数据集导入
'''
# @Time : 2021/6/3 10:40 
# @Author : LINYANZHEN
# @File : DTD.py
import pickle
import os
from PIL import Image
import numpy
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
import utils
import time
import scipy.io as scio


class DTDDataset(torch.utils.data.Dataset):
    def __init__(self, label_file_path, img_dir, img_transforms=None, loader=utils.image_loader):
        '''
        The data is split in three equal parts, in train, validation and test, 40 images per class, for each split.
        We provide the ground truth annotation for both key and joint attributes,
        as well as the 10 splits of the data we used for evaluation.

        一共47个种类，每种一共120张图片，被均匀分为3类，train、validation、test，每类40张图片
        一共有10种分法

        :param label_file_path: label文件路径，里面标明了被分配到train、validation、test的图片的路径
        :param img_dir: 所有图片的根路径
        :param img_transforms: 图片的预处理
        :param loader: 图片加载函数
        '''
        # 读取文件，获取图片路径
        image_path_list = []
        with open(label_file_path, "r") as file:
            image_path_list = file.readlines()
        self.image_path_list = []
        for i in image_path_list:
            # 去掉路径前后的各种空格
            i = i.strip()
            self.image_path_list.append(os.path.join(img_dir, i))
        self.label_list = []
        # 一共47个类别，每类都固定分配40张图片
        # 将原本的label转化为数字，方便运算
        for i in range(47):
            for j in range(40):
                self.label_list.append(i)
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
        print("数据集读取完毕")

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
    img_dir = "G:\\Dataset\\dtd\\images"
    train_file_path = "G:\\Dataset\\dtd\\labels\\train1.txt"
    test_file_path = "G:\\Dataset\\dtd\\labels\\test1.txt"
    val_file_path = "G:\\Dataset\\dtd\\labels\\val1.txt"

    train_loader = torch.utils.data.DataLoader(DTDDataset(train_file_path, img_dir))
    val_loader = torch.utils.data.DataLoader(DTDDataset(val_file_path, img_dir))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
