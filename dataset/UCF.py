# -*- coding: utf-8 -*-
'''
UCF数据集导入
'''
# @Time : 2021/6/6 17:04 
# @Author : LINYANZHEN
# @File : UCF.py
import pickle
import os
from PIL import Image
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
import utils
import time
import scipy.io as scio
import h5py

class UCFDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass