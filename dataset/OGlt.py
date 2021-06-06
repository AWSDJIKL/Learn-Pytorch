# -*- coding: utf-8 -*-
'''
OGlt数据集导入
'''
# @Time : 2021/6/6 14:52 
# @Author : LINYANZHEN
# @File : OGlt.py
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

class OGltDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir):
        pass
