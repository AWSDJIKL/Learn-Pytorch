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
import cv2
import random


def image_loader(image_path):
    return Image.open(image_path).convert('RGB')


def video_loader(video_path, frame_num=16):
    # print(video_path)
    capture = cv2.VideoCapture(video_path)
    # 获取视频总帧数
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_count)
    result = []
    # 有些视频的某些帧取不到，原因未知
    for i in random.sample(range(frame_count), frame_num):
        capture.set(1, i)
        rval, frame = capture.read()
        if not rval:
            print(video_path)
            print("frame_count=", frame_count)
            print("i", i)
            # 取不到的帧数量不多，先放弃
            continue
        # print(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        result.append(frame)
    return result


def prepare_dataloaders(dataset_list):
    train_loader_list = []
    val_loader_list = []
    num_classes_list = []
    for dataset in dataset_list:
        train_loader, val_loader, num_classes = get_dataloader(dataset)
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
    return train_loader_list, val_loader_list, num_classes_list


def get_dataloader(dataset_name):
    if dataset_name == "ImageNet":
        from dataset import ImageNet
        train_data_dir = "D:/Dataset/imagenet2012/ILSVRC2012_img_train"
        val_data_dir = "D:/Dataset/imagenet2012/ILSVRC2012_img_val"
        devkit_dir = "D:/Dataset/imagenet2012/ILSVRC2012_devkit_t12/ILSVRC2012_devkit_t12"
        batch_size = 128
        num_workers = 4

        train_loader = torch.utils.data.DataLoader(ImageNet.ImageNetDataset(train_data_dir, devkit_dir, mode="train"),
                                                   batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(ImageNet.ImageNetDataset(val_data_dir, devkit_dir, mode="validate"),
                                                 num_workers=num_workers, pin_memory=True)
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "Aircraft":
        from dataset import Aircraft
        data_dir = "D:/Dataset/fgvc-aircraft-2013b/data/images"
        train_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_variant_train.txt"
        val_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_variant_val.txt"
        all_labels = "D:/Dataset/fgvc-aircraft-2013b/data/variants.txt"
        batch_size = 16
        train_loader = DataLoader(Aircraft.AircraftDataset(data_dir, train_labels, all_labels), batch_size=batch_size,
                                  num_workers=4)
        val_loader = DataLoader(Aircraft.AircraftDataset(data_dir, val_labels, all_labels))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "CIFAR100":
        from dataset import CIFAR100
        train_file_path = "D:/Dataset/cifar-100-python/cifar-100-python/train"
        val_file_path = "D:/Dataset/cifar-100-python/cifar-100-python/test"
        label_file_path = "D:/Dataset/cifar-100-python/cifar-100-python/meta"
        train_loader = torch.utils.data.DataLoader(CIFAR100.CIFAR100Dataset(train_file_path))
        val_loader = torch.utils.data.DataLoader(CIFAR100.CIFAR100Dataset(val_file_path))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "DPED":
        from dataset import DPED
        train_file_path = "D:/Dataset/DPED/original_images/train"
        test_file_path = "D:/Dataset/DPED/original_images/test"
        # DPEDDataset(train_file_path)

        train_loader = torch.utils.data.DataLoader(DPED.DPEDDataset(train_file_path), num_workers=8)
        val_loader = torch.utils.data.DataLoader(DPED.DPEDDataset(test_file_path))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "DTD":
        from dataset import DTD
        img_dir = "D:/Dataset/dtd/images"
        train_file_path = "D:/Dataset/dtd/labels/train1.txt"
        test_file_path = "D:/Dataset/dtd/labels/test1.txt"
        val_file_path = "D:/Dataset/dtd/labels/val1.txt"

        train_loader = torch.utils.data.DataLoader(DTD.DTDDataset(train_file_path, img_dir))
        val_loader = torch.utils.data.DataLoader(DTD.DTDDataset(val_file_path, img_dir))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "GTSR":
        from dataset import GTSR
        train_file_path = "D:/Dataset/GTSRB/GTSRB/Final_Training/Images"
        test_file_path = "D:/Dataset/GTSRB/GTSRB/Final_Test/Images"
        csv_file_path = "D:/Dataset/GTSRB/GTSRB/Final_Test/GT-final_test.csv"

        train_loader = torch.utils.data.DataLoader(GTSR.GTSRDataset(train_file_path), pin_memory=True, num_workers=4,
                                                   batch_size=128)
        val_loader = torch.utils.data.DataLoader(GTSR.GTSRDataset(test_file_path, csv_path=csv_file_path, mode="test"))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "Flwr":
        from dataset import Flwr
        train_file_path = "D:/Dataset/oxford-102-flower-pytorch/flower_data/train"
        val_file_path = "D:/Dataset/oxford-102-flower-pytorch/flower_data/valid"
        test_file_path = "D:/Dataset/oxford-102-flower-pytorch/flower_data/test"

        train_loader = torch.utils.data.DataLoader(Flwr.FlwrDataset(train_file_path, mode="train"))
        val_loader = torch.utils.data.DataLoader(Flwr.FlwrDataset(val_file_path, mode="val"))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "SVHN":
        from dataset import SVHN
        train_file_path = "D:/Dataset/SVHN/train_32x32.mat"
        extra_file_path = "D:/Dataset/SVHN/extra_32x32.mat"
        test_file_path = "D:/Dataset/SVHN/test_32x32.mat"

        train_loader = torch.utils.data.DataLoader(SVHN.SVHNDataset(train_file_path))
        val_loader = torch.utils.data.DataLoader(SVHN.SVHNDataset(test_file_path))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "OGlt":
        from dataset import OGlt
        data_dir = "D:/Dataset/archive"
        OGlt.OGltDataset.shuffle_train_test(train_rate=0.7)
        train_loader = torch.utils.data.DataLoader(OGlt.OGltDataset(data_dir, mode="train"))
        val_loader = torch.utils.data.DataLoader(OGlt.OGltDataset(data_dir, mode="test"))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
    elif dataset_name == "UCF":
        from dataset import UCF
        train_file_path = "D:/Dataset/ucfTrainTestlist/trainlist01.txt"
        test_file_path = "D:/Dataset/ucfTrainTestlist/testlist01.txt"
        video_to_label = "D:/Dataset/ucfTrainTestlist/classInd.txt"
        video_dir = "D:/Dataset/UCF-101"
        # 截取帧保存路径
        train_image_save_path = "D:/Dataset/UCF-101_image/train"
        test_image_save_path = "D:/Dataset/UCF-101_image/test"
        train_loader = torch.utils.data.DataLoader(
            UCF.UCFDataset(video_dir, train_file_path, video_to_label, train_image_save_path, mode="train"))
        val_loader = torch.utils.data.DataLoader(
            UCF.UCFDataset(video_dir, test_file_path, video_to_label, test_image_save_path, mode="test"))
        num_classes = train_loader.dataset.num_classes
        return train_loader, val_loader, num_classes
