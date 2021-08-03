# -*- coding: utf-8 -*-
'''
加载imagenet数据集
'''
# @Time : 2021/5/31 13:45 
# @Author : LINYANZHEN
# @File : ImageNet.py

import os
import torch
import torch.utils.data
from torchvision import transforms
from scipy.io import loadmat
import classify_utils


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, devkit_dir, mode="train", img_transforms=None, loader=classify_utils.image_loader):
        '''

        :param data_dir: 图片数据路径
        :param devkit_dir: 开发工具包路径
        :param mode: 模式(train/validate/test)
        :param img_transforms: 数据增强
        :param loader: 加载图片的方式
        '''
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
                # label要从0开始
                tem_label_list[i] = int(labels[index]) - 1
            # 展开成一维的
            for i in range(len(tem_image_paths)):
                image_path_list.extend(tem_image_paths[i])
                label_list.extend([tem_label_list[i]] * len(tem_image_paths[i]))
        elif mode == "validate":
            for root, dirs, files in os.walk(data_dir):
                for i in files:
                    image_path_list.append(os.path.join(root, i))
            with open(os.path.join(devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt"), "r") as file:
                for line in file.readlines():
                    # 去掉空格
                    label_list.append(int(line.strip()))
            print(label_list)
        elif mode == "test":
            for root, dirs, files in os.walk(data_dir):
                for i in files:
                    image_path_list.append(os.path.join(root, i))
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
        # 分类类别数，用于最后的全连接分类
        self.num_classes = len(set(self.label_list))
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


def prepare_dataloader():
    train_data_dir = "G:/Dataset/imagenet2012/ILSVRC2012_img_train"
    test_data_dir = "G:/Dataset/imagenet2012\ILSVRC2012_img_val"
    devkit_dir = "G:\Dataset\imagenet2012\ILSVRC2012_devkit_t12"

    train_loader = torch.utils.data.DataLoader(ImageNetDataset(train_data_dir, devkit_dir, mode="train"))
    val_loader = torch.utils.data.DataLoader(ImageNetDataset(test_data_dir, devkit_dir, mode="validate"))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
