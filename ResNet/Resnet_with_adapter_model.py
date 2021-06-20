# models.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright © The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import config
import math


def conv3x3(in_planes, out_planes, stride=1):
    '''
    3*3卷积层，padding为1，即不改变图像大小

    :param in_planes: 输入通道
    :param out_planes: 输出通道
    :param stride: 步长
    :return:
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


class conv1x1(nn.Module):
    def __init__(self, planes, out_planes=None, stride=1):
        '''
        adapter，包含一个1*1卷积块和BN

        :param planes: 输入通道
        :param out_planes: 输出通道
        :param stride: 步长
        '''
        super(conv1x1, self).__init__()
        if config.mode == 'series_adapters':
            self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes))
        elif config.mode == 'parallel_adapters':
            self.conv = conv1x1_fonc(planes, out_planes, stride)
        else:
            self.conv = conv1x1_fonc(planes)

    def forward(self, x):
        y = self.conv(x)
        if config.mode == 'series_adapters':
            y += x
        return y


class conv_task(nn.Module):

    def __init__(self, in_planes, planes, stride=1, nb_tasks=1, is_proj=1, second=0):
        '''
        定义一个卷积层中的其中一个的卷积操作

        :param in_planes: 输入通道
        :param planes: 输出通道
        :param stride: 步长
        :param nb_tasks: 要训练的任务数量（即同时训练多少个数据集）
        :param is_proj: 是否使用adapter
        :param second: 指明是第一个卷积还是第二个卷积
        '''
        super(conv_task, self).__init__()
        self.is_proj = is_proj
        self.second = second
        self.conv = conv3x3(in_planes, planes, stride)
        if config.mode == 'series_adapters' and is_proj:
            self.bns = nn.ModuleList([nn.Sequential(conv1x1(planes), nn.BatchNorm2d(planes)) for i in range(nb_tasks)])
        elif config.mode == 'parallel_adapters' and is_proj:
            self.parallel_conv = nn.ModuleList([conv1x1(in_planes, planes, stride) for i in range(nb_tasks)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        else:
            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])

    def forward(self, x):
        # 记录当此训练的task
        task = config.task
        # 先经过卷积操作
        y = self.conv(x)
        # dropout
        if self.second == 0:
            if config.isdropout1:
                x = F.dropout2d(x, p=0.5, training=self.training)
        else:
            if config.isdropout2:
                x = F.dropout2d(x, p=0.5, training=self.training)
        if config.mode == 'parallel_adapters' and self.is_proj:
            # 只经过对应task的adapter
            y = y + self.parallel_conv[task](x)
        # 经过对应adapter的BN
        y = self.bns[task](y)

        return y


# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=0, nb_tasks=1, use_adapter=1):
        super(BasicBlock, self).__init__()
        if use_adapter == 1:
            # 使用
            self.conv1 = conv_task(in_planes, planes, stride, nb_tasks, is_proj=int(config.proj[0]))
            self.conv2 = nn.Sequential(nn.ReLU(True),
                                       conv_task(planes, planes, 1, nb_tasks, is_proj=int(config.proj[1]),
                                                 second=1))
        else:
            # 不使用
            self.conv1 = conv_task(in_planes, planes, stride, nb_tasks, is_proj=0)
            self.conv2 = nn.Sequential(nn.ReLU(True),
                                       conv_task(planes, planes, 1, nb_tasks, is_proj=0, second=1))
        self.shortcut = shortcut
        if self.shortcut == 1:
            self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        residual = x
        # 先2层卷积
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut == 1:
            # 平均池化，2*2
            residual = self.avgpool(x)
            # 横向拼接一个和x同样大小的全零矩阵
            residual = torch.cat((residual, residual * 0), 1)
        y += residual
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, block, nblocks, num_classes=[10]):
        super(ResNet, self).__init__()
        nb_tasks = len(num_classes)
        blocks = [block, block, block]
        factor = config.factor
        self.in_planes = int(32 * factor)
        # 最上层的3*3卷积
        self.pre_layers_conv = conv_task(3, int(32 * factor), 1, nb_tasks)

        # 记住需要使用的adapter
        mode = config.mode
        # 3个卷积层结构
        # 根据需要在前中后三个层次加入adapter
        self.layer1 = self._make_layer(blocks[0], int(64 * factor), nblocks[0], stride=2, nb_tasks=nb_tasks,
                                       use_adapter=int(config.ad_pos[0]))
        self.layer2 = self._make_layer(blocks[1], int(128 * factor), nblocks[1], stride=2, nb_tasks=nb_tasks,
                                       use_adapter=int(config.ad_pos[1]))
        self.layer3 = self._make_layer(blocks[2], int(256 * factor), nblocks[2], stride=2, nb_tasks=nb_tasks,
                                       use_adapter=int(config.ad_pos[2]))

        # 将第一个state输出变换到与第二个state输出相同大小
        # x1:(128,64,32,32) to x2:(128,128,16,16)
        self.x1_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1))
        # 将第一个state输出变换到与第三个state输出相同大小
        # x1:(128,64,32,32) to x2:(128,256,8,8)
        self.x1_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1))

        # BN
        self.end_bns = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm2d(int(256 * factor)), nn.ReLU(True)) for i in range(nb_tasks)])
        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 全连接
        self.linears = nn.ModuleList([nn.Linear(int(256 * factor), num_classes[i]) for i in range(nb_tasks)])
        # 遍历模型里所有的模块
        for m in self.modules():
            # 如果是卷积模块
            if isinstance(m, nn.Conv2d):
                # 根据参数量初始化权重，服从均值为0，方差为math.sqrt(2. / n)的正态分布
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # 如果是BN模块
            elif isinstance(m, nn.BatchNorm2d):
                # 权重全部初始化为1
                m.weight.data.fill_(1)
                # 偏置初始化为0
                m.bias.data.zero_()

    def _make_layer(self, block, planes, nblocks, stride=1, nb_tasks=1, use_adapter=1):
        shortcut = 0
        # 如果输入输出通道不一样或者卷积的步长不为1，则需要在里面多加一个平均池化
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1
        # 卷积结构里的层
        layers = []
        # 构建第一层，第一层需要单独构建，因为要加shortcut
        layers.append(block(self.in_planes, planes, stride, shortcut, nb_tasks=nb_tasks, use_adapter=use_adapter))
        self.in_planes = planes * block.expansion
        # 从1开始，因为上面已经有1层了
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, nb_tasks=nb_tasks, use_adapter=use_adapter))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers_conv(x)
        task = config.task

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        # 将x1变换到与x2相同大小
        x2 += self.x1_2(x1)

        x3 = self.layer3(x2)

        # 将x1变换到与x3相同大小
        x3 += self.x1_3(x1)

        x = x3

        x = self.end_bns[task](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linears[task](x)
        return x


def resnet26(num_classes=10, blocks=BasicBlock):
    return ResNet(blocks, [4, 4, 4], num_classes)


def resnet18(num_classes=10, blocks=BasicBlock):
    return ResNet(blocks, [4, 4, 4], num_classes)
