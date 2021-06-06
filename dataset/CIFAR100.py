# -*- coding: utf-8 -*-
'''
CIFAR100数据集导入
'''
# @Time : 2021/6/1 11:25 
# @Author : LINYANZHEN
# @File : CIFAR100.py
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


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, pickle_file_path, img_transforms=None):
        '''
        CIFAR100数据集

        :param pickle_file_path: pickle文件，里面装有图片矩阵、对应的数字标签和数字对应的具体标签
        :param img_transforms: 数据预处理方式
        '''
        image_and_label = unpickle(pickle_file_path)
        self.images = []
        # 这里的label已经是数字了
        self.labels = image_and_label[b"fine_labels"]
        data = image_and_label[b"data"]
        # 将一维数组变成3维数组，3通道，32*32
        for i in data:
            image = np.array(i)
            image = image.reshape((3, 32, 32))
            self.images.append(image)
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
        # print(self.images[0])
        # print(self.labels[0])

    def __getitem__(self, index):
        # 将数字转成图像
        # print(np.transpose(self.images[index], (1, 2, 0)))
        # Image.fromarray只接受(w,h,c)的格式，即通道要放最后
        img = Image.fromarray(np.transpose(self.images[index], (1, 2, 0)))
        # img.show()
        if self.img_transforms:
            img = self.img_transforms(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.images)


def prepare_dataloader():
    train_file_path = "G:\\Dataset\\cifar-100-python\\train"
    test_file_path = "G:\\Dataset\\cifar-100-python\\test"
    label_file_path = "G:\\Dataset\\cifar-100-python\\meta"
    train_loader = torch.utils.data.DataLoader(CIFAR100Dataset(train_file_path))
    val_loader = torch.utils.data.DataLoader(CIFAR100Dataset(test_file_path))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
