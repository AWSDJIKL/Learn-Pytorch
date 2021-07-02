# -*- coding: utf-8 -*-
'''
Aircraft数据集导入
'''
# @Time : 2021/5/31 15:24 
# @Author : LINYANZHEN
# @File : Aircraft.py
import os
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
import utils
import time


class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_and_label, all_labels, img_transforms=None,
                 loader=utils.image_loader):
        '''
        Aircraft数据集

        :param data_dir: 图片数据的根路径
        :param image_and_label:
        :param all_labels:
        :param mode:
        :param img_transforms:
        :param loader:
        '''
        self.image_path_list = []
        self.label_list = []
        self.loader = loader
        # 读取文件
        with open(image_and_label, "r") as file:
            image_and_label = file.readlines()
        with open(all_labels, "r") as file:
            all_labels = file.readlines()
        # 先将所有label转为数字
        all_labels = dict(zip(all_labels, range(len(all_labels))))
        # print(all_labels)
        # print(image_and_label)

        for i in image_and_label:
            self.image_path_list.append(os.path.join(data_dir, i[:7] + ".jpg"))
            self.label_list.append(all_labels[i[8:]])
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
    data_dir = "G:/Dataset/aircraft/fgvc-aircraft-2013b/data/images"
    train_labels = "G:/Dataset/aircraft/fgvc-aircraft-2013b/data/images_variant_train.txt"
    val_labels = "G:/Dataset/aircraft/fgvc-aircraft-2013b/data/images_variant_val.txt"
    all_labels = "G:/Dataset/aircraft/fgvc-aircraft-2013b/data/variants.txt"

    train_loader = torch.utils.data.DataLoader(AircraftDataset(data_dir, train_labels, all_labels))
    val_loader = torch.utils.data.DataLoader(AircraftDataset(data_dir, val_labels, all_labels))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return


if __name__ == '__main__':
    prepare_dataloader()
