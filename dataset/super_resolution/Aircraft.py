# -*- coding: utf-8 -*-
'''
Aircraft数据集导入(用于超分辨率)
飞机数据集本应作为分类数据集使用，这里仅用于测试
'''
# @Time    : 2021/7/14 22:40
# @Author  : LINYANZHEN
# @File    : Aircraft.py
import os
import torch
import torch.utils.data
from torchvision import transforms
import super_resolution_utils as utils
from PIL import Image


class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_list_txt, upscale_factor=3,
                 loader=utils.image_loader):
        '''
        Aircraft数据集

        :param data_dir:
        :param image_list_txt:
        :param upscale_factor:
        :param lr_transform:
        :param hr_transform:
        :param loader:
        '''
        self.image_path_list = []
        self.loader = loader
        self.upscale_factor = upscale_factor
        with open(image_list_txt, "r") as file:
            for i in file.readlines():
                i = i.strip()
                self.image_path_list.append(os.path.join(data_dir, i + ".jpg"))


    def __getitem__(self, index):
        path = self.image_path_list[index]
        # 加载原图
        original_img = self.loader(path)
        img_size = original_img.size
        # 避免因为除不尽导致生成图片与原图尺寸对不上
        img_size = [(i // self.upscale_factor) * self.upscale_factor for i in img_size]
        # print(img_size)
        hr_transform = utils.hr_transform(img_size=img_size)
        lr_transform = utils.lr_transform(img_size=img_size, upscale_factor=self.upscale_factor)
        # 对原图做高清图像的预处理，原图作为y
        original_img = hr_transform(original_img)
        # 使用预处理缩小模糊图片，作为x
        train_img = lr_transform(original_img)
        return train_img, original_img

    def __len__(self):
        return len(self.image_path_list)


def prepare_dataloader():
    data_dir = "D:/Dataset/fgvc-aircraft-2013b/data/images"
    train_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_train.txt"
    val_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_val.txt"

    train_loader = torch.utils.data.DataLoader(AircraftDataset(data_dir, train_labels))
    val_loader = torch.utils.data.DataLoader(AircraftDataset(data_dir, val_labels))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return


if __name__ == '__main__':
    prepare_dataloader()
