# -*- coding: utf-8 -*-
'''
把h5py文件里的图片取出来
'''
# @Time    : 2021/7/24 14:56
# @Author  : LINYANZHEN
# @File    : get_h5py_file.py

import h5py
import numpy as np
from PIL import Image


def get_narray(h5_file):
    lr_list = []
    hr_list = []
    with h5py.File(h5_file, 'r') as f:
        for i in range(len(f['lr'])):
            lr_list.append(f['lr'][i] )
            hr_list.append(f['hr'][i] )
    return lr_list, hr_list


def narray_to_image(narray):
    print(narray.shape)
    img = Image.fromarray(np.uint8(narray), mode='L')
    img.show()


h5_file = "D:/Dataset/ESCPN/91-image_x3.h5"
a, b = get_narray(h5_file)
narray_to_image(b[1])
