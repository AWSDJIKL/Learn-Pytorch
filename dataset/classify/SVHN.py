# -*- coding: utf-8 -*-
'''
SVHN数据集导入
'''
# @Time : 2021/6/6 16:07 
# @Author : LINYANZHEN
# @File : SVHN.py

from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms
import scipy.io as scio


class SVHNDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, img_transforms=None):
        '''
        Character level ground truth in an MNIST-like format.
        All digits have been resized to a fixed resolution of 32-by-32 pixels.
        The original character bounding boxes are extended in the appropriate dimension to become square windows,
        so that resizing them to 32-by-32 pixels does not introduce aspect ratio distortions.
        Nevertheless this preprocessing introduces some distracting digits to the sides of the digit of interest.
        Loading the .mat files creates 2 variables: X which is a 4-D matrix containing the images,
        and y which is a vector of class labels.
        To access the images, X(:,:,:,i) gives the i-th 32-by-32 RGB image, with class label y(i).

        :param data_file: 数据文件路径
        :param img_transforms: 图片预处理方式
        '''
        self.image_list = []
        self.label_list = []
        data = load_mat_file(data_file)
        x = data["X"]
        y = data["y"]
        # print(x.shape)  # (32, 32, 3, 73257)
        # print(y.shape)  # (73257, 1)
        for i in range(x.shape[-1]):
            self.image_list.append(x[:, :, :, i])
            # 标签要从0开始
            self.label_list.append(int(y[i]) - 1)
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
        print(self.label_list)
        # 分类类别数，用于最后的全连接分类
        self.num_classes = len(set(self.label_list))

    def __getitem__(self, index):
        # 将数字转成图像
        img = Image.fromarray(self.image_list[index])
        # img.show()
        if self.img_transforms:
            img = self.img_transforms(img)
        return img, self.label_list[index]

    def __len__(self):
        return len(self.image_list)


def load_mat_file(file_path):
    return scio.loadmat(file_path)


def prepare_dataloader():
    train_file_path = "G:\\Dataset\\SVHN\\train_32x32.mat"
    extra_file_path = "G:\\Dataset\\SVHN\\extra_32x32.mat"
    test_file_path = "G:\\Dataset\\SVHN\\test_32x32.mat"

    train_loader = torch.utils.data.DataLoader(SVHNDataset(train_file_path))
    val_loader = torch.utils.data.DataLoader(SVHNDataset(test_file_path))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return train_loader, val_loader


if __name__ == '__main__':
    prepare_dataloader()
    # train_file = "G:\\Dataset\\SVHN\\train_32x32.mat"
    # load_mat_file(train_file)
