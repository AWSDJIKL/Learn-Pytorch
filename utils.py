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

        train_loader = DataLoader(AircraftDataset.AircraftDataset(data_dir, train_labels, all_labels))
        val_loader = DataLoader(AircraftDataset.AircraftDataset(data_dir, val_labels, all_labels))
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
    elif dataset_name == "DTD":
        from dataset import DTD
        img_dir = "G:\\Dataset\\dtd\\images"
        train_file_path = "G:\\Dataset\\dtd\\labels\\train1.txt"
        test_file_path = "G:\\Dataset\\dtd\\labels\\test1.txt"
        val_file_path = "G:\\Dataset\\dtd\\labels\\val1.txt"

        train_loader = torch.utils.data.DataLoader(DTD.DTDDataset(train_file_path, img_dir))
        val_loader = torch.utils.data.DataLoader(DTD.DTDDataset(val_file_path, img_dir))
        return train_loader, val_loader
    elif dataset_name == "GTSR":
        from dataset import GTSR
        train_file_path = "G:\\Dataset\\GTSRB\\GTSRB\\Final_Training\\Images"
        test_file_path = "G:\\Dataset\\GTSRB\\GTSRB\\Final_Test\\Images"
        csv_file_path = "G:\\Dataset\\GTSRB\\GTSRB\\Final_Test\\GT-final_test.csv"

        train_loader = torch.utils.data.DataLoader(GTSR.GTSRDataset(train_file_path))
        val_loader = torch.utils.data.DataLoader(GTSR.GTSRDataset(test_file_path, csv_path=csv_file_path, mode="test"))
        return train_loader, val_loader
    elif dataset_name == "Flwr":
        from dataset import Flwr
        train_file_path = "G:\\Dataset\\oxford-102-flower-pytorch\\flower_data\\train"
        val_file_path = "G:\\Dataset\\oxford-102-flower-pytorch\\flower_data\\valid"
        test_file_path = "G:\\Dataset\\oxford-102-flower-pytorch\\flower_data\\test"

        train_loader = torch.utils.data.DataLoader(Flwr.FlwrDataset(train_file_path, mode="train"))
        val_loader = torch.utils.data.DataLoader(Flwr.FlwrDataset(val_file_path, mode="val"))
        return train_loader, val_loader
    elif dataset_name == "SVHN":
        from dataset import SVHN
        train_file_path = "G:\\Dataset\\SVHN\\train_32x32.mat"
        extra_file_path = "G:\\Dataset\\SVHN\\extra_32x32.mat"
        test_file_path = "G:\\Dataset\\SVHN\\test_32x32.mat"

        train_loader = torch.utils.data.DataLoader(SVHN.SVHNDataset(train_file_path))
        val_loader = torch.utils.data.DataLoader(SVHN.SVHNDataset(test_file_path))
        return train_loader, val_loader
