# -*- coding: utf-8 -*-
'''
CNN模型训练
'''
# @Time : 2021/6/7 19:38 
# @Author : LINYANZHEN
# @File : CNN_train.py
import SimpleCNN
import time

import torch
import torch.nn as nn
import classify_utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


def train(model, train_loader, criterion, optimizer, epoch):
    loss_list = []
    for i in range(epoch):
        epoch_start_time = time.time()
        print("epoch[{}/{}]".format(i, epoch))
        for index, (x, y) in enumerate(train_loader, 0):
            x = x.to(device)
            y = y.to(device)
            out = model(x).squeeze(-1)
            optimizer.zero_grad()
            # out = out.view(17, 64)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))
    # 画出损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, "b")
    plt.title('loss')
    plt.savefig('Temperature_Train_loss_RNN_with_GPU.jpg', dpi=256)
    plt.close()


def test(model, test_loader):
    y_pred_list = []
    y_true_list = []
    for i, (x, y) in enumerate(test_loader, 0):
        x = x.to(device)
        y = y.to(device)
        # 计算预测值
        y_pred = model(x)
        y_pred = y_pred.detach().cpu().numpy().reshape(y_pred.shape[0])
        y = y.detach().cpu().numpy().reshape(y.shape[0])
        if i == 0:
            y_pred_list = y_pred
            y_true_list = y
        else:
            y_pred_list = np.concatenate((y_pred_list, y_pred))
            y_true_list = np.concatenate((y_true_list, y))
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_list, 'r+', label='real data')
    plt.plot(y_pred_list, 'b*', label='pred data')
    plt.legend()
    plt.savefig('pre_temperature_RNN_with_GPU.jpg', dpi=256)
    plt.close()


if __name__ == '__main__':
    start_time = time.time()
    train_loader, test_loader = classify_utils.get_classify_dataloader("UCF")

    device = torch.device('cuda:0')
    model = SimpleCNN.SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
    criterion = nn.CrossEntropyLoss()
    epoch = 10

    # 训练模型
    train(model, train_loader, criterion, optimizer, epoch)
    # 保存模型
    print("训练完成")
    torch.save(model, "temperature_model_RNN_with_GPU.pth")
    print("模型已保存")

    # 测试模型
    model = torch.load("temperature_model_RNN_with_GPU.pth")
    test(model, test_loader)
    print("总耗时:{}min".format((time.time() - start_time) / 60))
