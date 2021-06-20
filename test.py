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
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
from PIL import Image
import time
import utils
import cv2

if __name__ == '__main__':
    l = list(range(1, 21))
    random.shuffle(l)
    print(l)
    train = [l.pop() for i in range(int(20 * 0.7))]
    print(train)
    test = l

    print(test)
