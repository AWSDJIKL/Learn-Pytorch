# -*- coding: utf-8 -*-
'''
超分辨的工具函数
'''
# @Time    : 2021/7/15 17:37
# @Author  : LINYANZHEN
# @File    : super_resolution_utils.py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import random
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
from torchvision.transforms import ToTensor


def calaculate_psnr(img1, img2):
    '''
    计算两张图片之间的PSNR误差

    :param img1:
    :param img2:
    :return:
    '''
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2)).item()


def image_loader(image_path):
    '''

    :param image_path:
    :return:
    '''
    y, cb, cr = Image.open(image_path).convert('YCbCr').split()
    y = Variable(ToTensor()(y))
    cb = Variable(ToTensor()(cb))
    cr = Variable(ToTensor()(cr))
    return y, cb, cr


def lr_transform(img_size, upscale_factor):
    '''

    :param img_size: 原图大小
    :param upscale_factor:
    :return:
    '''
    new_size = [i // upscale_factor for i in img_size]
    # print("original size =", img_size)
    # print("new size =", new_size)
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.Resize(new_size, interpolation=InterpolationMode.BICUBIC),
        # transforms.ToTensor(),
    ])


def hr_transform(img_size):
    '''

    :param img_size:
    :return:
    '''
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def tensor_to_image(tensor):
    return transforms.ToPILImage()(tensor)


class AverageMeter(object):
    '''
    记录数据并计算平均数
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


def prepare_super_resolution_loaders(dataset_list):
    train_loader_list = []
    val_loader_list = []
    for dataset in dataset_list:
        train_loader, val_loader = get_super_resolution_dataloader(dataset)
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
    return train_loader_list, val_loader_list


def get_super_resolution_dataloader(dataset_name):
    if dataset_name == "Aircraft":
        from dataset.super_resolution import Aircraft_ycbcr
        train_lr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/train/lr"
        train_hr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/train/hr"
        test_lr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/test/lr"
        test_hr_dir = "D:/Dataset/fgvc-aircraft-2013b/data/test/hr"

        train_loader = torch.utils.data.DataLoader(Aircraft_ycbcr.AircraftDataset(train_lr_dir, train_hr_dir))
        val_loader = torch.utils.data.DataLoader(Aircraft_ycbcr.AircraftDataset(test_lr_dir, test_hr_dir))

        return train_loader, val_loader
    elif dataset_name == "Aircraft_h5py":
        from dataset.super_resolution import Aircraft_h5py
        train_data_path = "D:/GitHub/Learn-Pytorch/dataset/super_resolution/aircraft_x3.h5"
        val_data_path = "D:/GitHub/Learn-Pytorch/dataset/super_resolution/aircraft_testset_x3.h5"
        batch_size = 160
        num_workers = 6
        train_loader = torch.utils.data.DataLoader(Aircraft_h5py.TrainDataset(h5_file=train_data_path),
                                                   batch_size=batch_size, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(Aircraft_h5py.EvalDataset(h5_file=val_data_path))
        return train_loader, val_loader
