# -*- coding: utf-8 -*-
'''
一个简易的CNN模型，主要用于测试训练集导入是否有问题
'''
# @Time : 2021/6/7 18:32 
# @Author : LINYANZHEN
# @File : SimpleCNN.py
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(kernel_size=3, stride=1, padding=1))
        self.fc = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1)
        x = self.fc(x)
        return x
