# -*- coding: utf-8 -*-
'''
临时代码单元测试
'''
# @Time : 2021/4/12 14:35 
# @Author : LINYANZHEN
# @File : test.py
import os
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
from PIL import Image
import time


# mat_path = "dataset/ILSVRC2012_devkit_t12/data/meta.mat"
# m = loadmat(mat_path)
# # print(m["synsets"])
# print("总长度：", len(m["synsets"]))
# for i in m["synsets"]:
#     print(i["WNID"])
#     print("-" * 10)


def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, devkit_dir, mode="train", img_transforms=None, loader=pil_loader):
        image_path_list = []
        label_list = []
        if mode == "train":
            tem_image_paths = []
            tem_label_list = []
            # 将所有训练集的图片都加载到列表里
            for root, dirs, files in os.walk(data_dir):
                images = []
                for i in files:
                    images.append(os.path.join(root, i))
                if len(images) == 0:
                    continue
                tem_image_paths.append(images)
                # 文件夹名字为WNID
                l = root.replace(data_dir, "")
                tem_label_list.append(l[1:])
            # 读取meta文件，取出WNID和对应的分类ID
            mat_path = os.path.join(devkit_dir, "data", "meta.mat")
            m = loadmat(mat_path)
            wnid = [i[0][0] for i in m["synsets"][:]["WNID"]]
            labels = [i[0][0] for i in m["synsets"][:]["ILSVRC2012_ID"]]
            # 将WNID转换为分类ID
            for i in range(len(tem_label_list)):
                index = wnid.index(tem_label_list[i])
                tem_label_list[i] = labels[index]
            # 展开成一维的
            for i in range(len(tem_image_paths)):
                image_path_list.extend(tem_image_paths[i])
                label_list.extend([tem_label_list[i][0]] * len(tem_image_paths[i]))
        elif mode == "test":
            for root, dirs, files in os.walk(data_dir):
                for i in files:
                    image_path_list.append(os.path.join(root, i))
            with open(os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt"), "r") as file:
                for line in file.readlines():
                    # 去掉空格
                    label_list.append(line.strip())
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
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.loader = loader

        print("ImageNet读取完成")

    def __getitem__(self, index):
        path = self.image_path_list[index]
        label = torch.tensor(self.label_list[index], dtype=torch.long)
        img = self.loader(path)
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        return img, label

    def __len__(self):
        return len(self.image_path_list)


def prepare_imagenet_dataloader():
    train_data_dir = "F:/Dataset/imagenet2012/ILSVRC2012_img_train"
    test_data_dir = "F:/Dataset/imagenet2012\ILSVRC2012_img_val"
    devkit_dir = "dataset/ILSVRC2012_devkit_t12"

    train_loader = torch.utils.data.DataLoader(ImageNetDataset(train_data_dir, devkit_dir, "train"))
    test_loader = torch.utils.data.DataLoader(ImageNetDataset(test_data_dir, devkit_dir, "test"))
    return


if __name__ == '__main__':
    # data_dir = "dataset"
    # lable_dir = "dataset/ILSVRC2012_devkit_t12/data"
    # prepare_imagenet_dataloader(data_dir)
    # mat_path = "dataset/ILSVRC2012_devkit_t12/data/meta.mat"
    # m = loadmat(mat_path)
    # print(m["synsets"][:]["WNID"])
    # print([i[0][0] for i in m["synsets"][:]["WNID"]])
    # for i in m["synsets"][:1000]:
    #     print(i)
    #     print("WNID", i["WNID"][0])
    #     print("ILSVRC2012_ID", i["ILSVRC2012_ID"][0])
    #     print("words", i["words"][0])
    #     print("-" * 10)
    start_time = time.time()
    train_data_dir = "F:/Dataset/imagenet2012/ILSVRC2012_img_train"
    devkit_dir = "dataset/ILSVRC2012_devkit_t12"
    train_dataset = ImageNetDataset(train_data_dir, devkit_dir, "train")
    print(train_dataset[0])
    print(train_dataset.__len__())
    end_time = time.time()
    print("用时：{}min".format((end_time - start_time) / 60))
