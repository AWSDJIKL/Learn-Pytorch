# -*- coding: utf-8 -*-
'''
临时代码单元测试
'''
# @Time : 2021/4/12 14:35 
# @Author : LINYANZHEN
# @File : test.py
import os
import random

import torch
import super_resolution_utils as utils

import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision import transforms


class PixelShuffle_test(nn.Module):
    def __init__(self, upscale_factor):
        '''
        亚像素卷积网络

        :param upscale_factor: 放大倍数
        '''
        super(PixelShuffle_test, self).__init__()
        # 重新排列像素
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.pixel_shuffle(x)
        return x


if __name__ == '__main__':
    img_path = "Sub-pixel-convolution/test/y.jpg"
    img = Image.open(img_path).convert('YCbCr')
    y, cb, cr = img.split()
    # 转tensor
    y = Variable(ToTensor()(y))
    cb = Variable(ToTensor()(cb))
    cr = Variable(ToTensor()(cr))
    # 重新转为img
    y = transforms.ToPILImage()(y)
    cb = transforms.ToPILImage()(cb)
    cr = transforms.ToPILImage()(cr)
    out_img = Image.merge('YCbCr', [y, cb, cr]).convert('RGB')
    out_img.show()
