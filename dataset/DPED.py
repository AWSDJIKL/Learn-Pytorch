# -*- coding: utf-8 -*-
'''
DPED数据集导入
'''
# @Time : 2021/6/2 20:15 
# @Author : LINYANZHEN
# @File : DPED.py
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


class DPEDDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_transforms=None, loader=utils.image_loader):
        '''
        DPED数据集

        图片已经按照类别放在对应的文件夹内

        :param data_dir:图片文件根路径
        :param img_transforms: 图片与处理方式
        :param loader: 图片加载函数
        '''
        self.image_path_list = []
        self.label_list = []
        self.loader = loader
        # 获取所有图片
        label = 0
        for root, dirs, files in os.walk(data_dir):
            for i in files:
                self.image_path_list.append(os.path.join(root, i))
                # 将label转化为数字
                # print(label)
                self.label_list.append(label)
                # print(label)
            if len(files)>0:
                label += 1
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
        # 分类类别数，用于最后的全连接分类
        self.num_classes = len(set(self.label_list))
        print("数据集读取完毕")

    def __getitem__(self, index):
        path = self.image_path_list[index]
        label = torch.tensor(self.label_list[index], dtype=torch.long)
        img = self.loader(path)
        if self.img_transforms:
            img = self.img_transforms(img)
        return img, label

    def __len__(self):
        return len(self.image_path_list)


def prepare_dataloader():
    train_file_path = "G:\\Dataset\\DPED\\original_images\\train"
    test_file_path = "G:\\Dataset\\DPED\\original_images\\test"
    # DPEDDataset(train_file_path)
    train_loader = torch.utils.data.DataLoader(DPEDDataset(train_file_path))
    val_loader = torch.utils.data.DataLoader(DPEDDataset(test_file_path))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
