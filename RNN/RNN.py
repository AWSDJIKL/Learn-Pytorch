# -*- coding: utf-8 -*-
'''
循环神经网络
'''
# @Time : 2021/4/19 17:06 
# @Author : LINYANZHEN
# @File : RNN.py


import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

dataset = torch.randn(seq_len, batch_size, input_size)
print(dataset)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print("Input size:", input.shape)
    hidden = cell(input, hidden)
    print("output size:", hidden.shape)
    print(hidden)
