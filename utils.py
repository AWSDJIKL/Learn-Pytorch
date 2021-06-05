# -*- coding: utf-8 -*-
'''
工具函数
'''
# @Time : 2021/5/31 16:06 
# @Author : LINYANZHEN
# @File : utils.py

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def pil_loader(path):
    return Image.open(path).convert('RGB')


def get_dataloader(dataset_name):
    if dataset_name == "ImageNet":
        from dataset import ImageNet
        train_data_dir = "F:/Dataset/imagenet2012/ILSVRC2012_img_train"
        val_data_dir = "F:/Dataset/imagenet2012\ILSVRC2012_img_val"
        devkit_dir = "dataset/ILSVRC2012_devkit_t12"

        train_loader = torch.utils.data.DataLoader(ImageNet.ImageNetDataset(train_data_dir, devkit_dir, mode="train"))
        val_loader = torch.utils.data.DataLoader(ImageNet.ImageNetDataset(val_data_dir, devkit_dir, mode="validate"))
        return train_loader, val_loader
    elif dataset_name == "Aircraft":
        from dataset import AircraftDataset
        data_dir = "F:/Dataset/aircraft/fgvc-aircraft-2013b/data/images"
        train_labels = "F:/Dataset/aircraft/fgvc-aircraft-2013b/data/images_variant_train.txt"
        val_labels = "F:/Dataset/aircraft/fgvc-aircraft-2013b/data/images_variant_val.txt"
        all_labels = "F:/Dataset/aircraft/fgvc-aircraft-2013b/data/variants.txt"

        train_loader = DataLoader(AircraftDataset.AircraftDataset(data_dir, train_labels, all_labels, mode="train"))
        val_loader = DataLoader(AircraftDataset.AircraftDataset(data_dir, val_labels, all_labels, mode="validate"))
        return train_loader, val_loader
    elif dataset_name == "CIFAR100":
        from dataset import CIFAR100
        train_file_path = "F:\\Dataset\\cifar-100-python\\train"
        val_file_path = "F:\\Dataset\\cifar-100-python\\test"
        label_file_path = "F:\\Dataset\\cifar-100-python\\meta"
        train_loader = torch.utils.data.DataLoader(CIFAR100.CIFAR100Dataset(train_file_path))
        val_loader = torch.utils.data.DataLoader(CIFAR100.CIFAR100Dataset(val_file_path))
        return train_loader, val_loader
    elif dataset_name == "DPED":
        from dataset import DPED
        train_file_path = "F:\\Dataset\\DPED\\original_images\\train"
        test_file_path = "F:\\Dataset\\DPED\\original_images\\test"
        # DPEDDataset(train_file_path)
        train_loader = torch.utils.data.DataLoader(DPED.DPEDDataset(train_file_path))
        val_loader = torch.utils.data.DataLoader(DPED.DPEDDataset(test_file_path))
        return train_loader, val_loader
