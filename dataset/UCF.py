# -*- coding: utf-8 -*-
'''
UCF数据集导入
'''
# @Time : 2021/6/6 17:04 
# @Author : LINYANZHEN
# @File : UCF.py
import pickle
import os
from PIL import Image
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
import utils
import time
import scipy.io as scio
import h5py


class UCFDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, video_path_txt, video_to_label, image_save_path, mode="train", img_transforms=None,
                 image_loader=utils.image_loader, video_loader=utils.video_loader, frame_count=16):
        start_time = time.time()
        video_path_list = []
        label_list = []
        # 读取txt文件，获取对应的视频文件地址
        if mode == "train":
            with open(video_path_txt, "r") as file:
                # 每一行是路径+数字标签
                for i in file.readlines():
                    i = i.split()
                    video_path_list.append(os.path.join(data_dir, i[0]))
                    label_list.append(int(i[1]))
        elif mode == "test":
            label_dict = {}
            with open(video_to_label, "r")as file:
                for i in file.readlines():
                    i = i.split()
                    label_dict[i[1]] = int(i[0])
            with open(video_path_txt, "r")as file:
                for i in file.readlines():
                    video_path_list.append(os.path.join(data_dir, i))
                    label_list.append(label_dict[os.path.split(i)[0]])
        self.image_path_list = []
        self.label_list = []
        # 统一路径格式
        for i in range(len(video_path_list)):
            video_path_list[i] = video_path_list[i].strip()
            video_path_list[i] = video_path_list[i].replace("\\", "/")
        # 将所有视频都随机抽取16帧并存放到指定位置
        if os.path.exists(image_save_path):
            # 路径已存在，清空里面的内容
            for root, dirs, files in os.walk(image_save_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        else:
            os.makedirs(image_save_path)
        # 计数器，用于图片的命名
        count = 0
        for vidoe_path, label in zip(video_path_list, label_list):
            # print(vidoe_path)
            # print(label)
            frames = video_loader(vidoe_path, frame_count)
            # 将图片保存起来
            for i in range(len(frames)):
                # 命名规则：第count个视频_第i个保存帧_label.jpg
                image_path = os.path.join(image_save_path, str(count) + "_" + str(i) + "_" + str(label) + ".jpg")
                frames[i].save(image_path)
                self.image_path_list.append(image_path)
                self.label_list.append(label)
            count += 1
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
        self.loader = image_loader
        end_time = time.time()
        print("截取帧保存完毕，用时{}min".format((end_time - start_time) / 60))

    def __getitem__(self, index):
        path = self.image_path_list[index]
        label = torch.tensor(self.label_list[index], dtype=torch.long)
        img = self.loader(path)
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        print(self.image_path_list[index])
        print(label)
        return img, label

    def __len__(self):
        return len(self.image_path_list)


def prepare_dataloader():
    train_file_path = "G:/Dataset/ucfTrainTestlist/trainlist01.txt"
    test_file_path = "G:/Dataset/ucfTrainTestlist/testlist01.txt"
    video_to_label = "G:/Dataset/ucfTrainTestlist/classInd.txt"
    video_dir = "G:/Dataset/UCF-101"
    train_image_save_path = "G:/Dataset/UCF-101_image/train"
    test_image_save_path = "G:/Dataset/UCF-101_image/test"
    train_loader = torch.utils.data.DataLoader(UCFDataset(video_dir, train_file_path, video_to_label, train_image_save_path, mode="train"))
    val_loader = torch.utils.data.DataLoader(
        UCFDataset(video_dir, test_file_path, video_to_label, test_image_save_path, mode="test"))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
    # train_file = "G:\\Dataset\\SVHN\\train_32x32.mat"
    # load_mat_file(train_file)
