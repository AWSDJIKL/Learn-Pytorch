# -*- coding: utf-8 -*-
'''
LeNet模型
'''
# @Time : 2021/4/11 12:18 
# @Author : LINYANZHEN
# @File : LeNetModule.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(16, 5, 5))
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.conv3(x)
        x = self.maxpooling(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
