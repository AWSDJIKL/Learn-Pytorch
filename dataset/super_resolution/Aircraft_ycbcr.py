# -*- coding: utf-8 -*-
'''
Aircraft数据集导入(用于超分辨率)
不使用常规RGB导入，使用ycbcr方式，训练时只使用y
'''
# @Time    : 2021/7/24 15:22
# @Author  : LINYANZHEN
# @File    : Aircraft_ycbcr.py
import os
import torch
import torch.utils.data
import super_resolution_utils as utils
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor


class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, loader=utils.image_loader):
        '''
        Aircraft数据集

        :param data_dir:
        :param image_list_txt:
        :param upscale_factor:
        :param lr_transform:
        :param hr_transform:
        :param loader:
        '''
        self.hr_list = []
        self.lr_list = []
        self.loader = loader
        for root, dirs, files in os.walk(hr_dir):
            for img in files:
                self.hr_list.append(os.path.join(hr_dir, img))
                self.lr_list.append(os.path.join(lr_dir, img))
        self.lr_list = self.lr_list[:100]
        self.hr_list = self.hr_list[:100]
        print("finsh")

    def __getitem__(self, index):
        # 加载图片
        lr_y, lr_cb, lr_cr = self.loader(self.lr_list[index])
        hr_y, hr_cb, hr_cr = self.loader(self.hr_list[index])

        return lr_y, lr_cb, lr_cr, hr_y, hr_cb, hr_cr

    def __len__(self):
        return len(self.hr_list)


def prepare_dataloader():
    data_dir = "D:/Dataset/fgvc-aircraft-2013b/data/images"
    train_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_train.txt"
    val_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_val.txt"

    train_loader = torch.utils.data.DataLoader(AircraftDataset(data_dir, train_labels))
    val_loader = torch.utils.data.DataLoader(AircraftDataset(data_dir, val_labels))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return


def prepare(data_dir, image_list_txt, lr_output_dir, hr_output_dir, upscale_factor=3, loader=utils.image_loader):
    # 先加载所有图片路径
    image_path_list = []
    with open(image_list_txt, "r") as file:
        for i in file.readlines():
            i = i.strip()
            image_path_list.append(os.path.join(data_dir, i + ".jpg"))
    print("图片路径加载完成")
    if not os.path.exists(lr_output_dir):
        os.makedirs(lr_output_dir)
    if not os.path.exists(hr_output_dir):
        os.makedirs(hr_output_dir)
    print("开始读取图片并生成hr、lr版本")
    # 读取图片，生成hr版本和lr版本并保存到指定路径
    count = 0
    for path in image_path_list:
        original_img = loader(path)
        img_size = original_img.size
        # 避免因为除不尽导致生成图片与原图尺寸对不上
        img_size = [(i // upscale_factor) * upscale_factor for i in img_size]
        # 先调整图片大小
        hr = original_img.resize((img_size[0], img_size[1]), resample=Image.BICUBIC)
        lr = original_img.resize((img_size[0] // upscale_factor, img_size[1] // upscale_factor), resample=Image.BICUBIC)
        hr.save(os.path.join(hr_output_dir, str(count) + ".jpg"))
        lr.save(os.path.join(lr_output_dir, str(count) + ".jpg"))
        count += 1
    print("finsh")


if __name__ == '__main__':
    data_dir = "D:/Dataset/fgvc-aircraft-2013b/data/images"
    train_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_train.txt"
    val_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_val.txt"
    train_lr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/train/lr"
    train_hr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/train/hr"
    test_lr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/test/lr"
    test_hr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/test/hr"
    prepare(data_dir, train_labels, train_lr_dir, train_hr_dir)
    prepare(data_dir, val_labels, test_lr_dir, test_hr_dir)
