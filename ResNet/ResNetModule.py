# -*- coding: utf-8 -*-
'''
ResNet
'''
# @Time : 2021/4/11 16:47 
# @Author : LINYANZHEN
# @File : ResNetModule.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # 先做一次7*7卷积，步长为2，加上3*3的最大池化，步长为2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        '''

        :param planes: 最后输出通道熟
        :param num_blocks: 残差结构块数量
        :param stride: 步长
        :return:
        '''
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.mp(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 平均池化
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 残差结构块，由2层卷积组成
class BasicBlock(nn.Module):  # tow 3*3 filters conv

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 卷积核步长不为1，输入输出通道数不一样，则做一次1*1卷积
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                # 若1*1卷积，为了改变输出通道数，因为后面要与输入相加，通道数上要匹配
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # 卷积+BN归一化
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 返回输出前加上输入
        out += self.shortcut(x)
        # ReLu函数激活
        out = F.relu(out)
        return out
