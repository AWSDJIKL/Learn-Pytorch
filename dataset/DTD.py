# -*- coding: utf-8 -*-
'''
DTD数据集导入
'''
# @Time : 2021/6/3 10:40 
# @Author : LINYANZHEN
# @File : DTD.py
import pickle
import os
from PIL import Image
import numpy
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
import utils
import time
import scipy.io as scio

dataFile = 'F:\\Dataset\\dtd\\imdb\\imdb.mat'
data = scio.loadmat(dataFile)
# print(data)
print(data.get("test"))
# print(type(data['meta'][0][0]))
