# -*- coding: utf-8 -*-
'''
VGGNet模型
'''
# @Time : 2021/4/11 15:30 
# @Author : LINYANZHEN
# @File : VGGNetModule.py


import torch
import torch.nn as nn

# 对应不同层数的VGGNet的结构
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'FC': [512 * 7 * 7, 4096, 4096, 10]
}


class VGGNet(nn.Module):
    def __init__(self, mode="VGG16", input_channels=3):
        '''

        :param mode: 要使用多深的VGGNet
        :param input_channels: 初始输入通道数
        '''
        super(VGGNet, self).__init__()
        # 设置卷积池化部分
        self.convs = nn.ModuleList()
        for output_channels in cfg[mode]:
            if output_channels == "M":
                # 是池化层
                self.convs.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.convs.append(nn.ReLU())
            else:
                self.convs.append(nn.Conv2d(input_channels, output_channels, kernel_size=3))
                self.convs.append(nn.ReLU())
                input_channels = output_channels
        # 设置全连接层部分
        self.fcs = nn.ModuleList()
        for i in range(len(cfg["FC"]) - 1):
            self.fcs.append(nn.Linear(cfg["FC"][i], cfg["FC"][i + 1]))
            self.fcs.append(nn.ReLU())

    def forward(self, x):
        for m in self.convs:
            x = m(x)
        x = x.view(x.shape[0], -1)
        for m in self.fcs:
            x = m(x)
        return x
