# -*- coding: utf-8 -*-
'''
LeNet训练
'''
# @Time : 2021/4/11 13:31 
# @Author : LINYANZHEN
# @File : LeNetTrain.py


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


def train(model, train_loader, test_loader, criterion, optimizer, epoch):
    pass


if __name__ == '__main__':
    train_set = datasets.MNIST(root="dataset/mnist", train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(root="dataset/mnist", train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False)
