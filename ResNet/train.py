# -*- coding: utf-8 -*-
'''
训练
'''
# @Time : 2021/6/18 11:00 
# @Author : LINYANZHEN
# @File : train.py

import Resnet_with_adapter_model as resnet
import time

import torch
import torch.nn as nn
import utils
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
            out = model(x)
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
    plt.savefig('resnet_with_adapter_loss.jpg', dpi=256)
    plt.close()


def test(model, test_loader):
    y_pred_list = []
    y_true_list = []
    true_count = 0
    for i, (x, y) in enumerate(test_loader, 0):
        x = x.to(device)
        y = y.to(device)
        # 计算预测值
        y_pred = model(x)
        # print(y)
        # print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)
        if y == predicted:
            true_count += 1
    print("acc={}".format(true_count / len(test_loader)))


if __name__ == '__main__':
    start_time = time.time()
    train_loader, test_loader = utils.get_dataloader("OGlt")
    num_classes = [train_loader.dataset.num_classes]
    device = torch.device('cuda:0')
    model = resnet.resnet18(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
    criterion = nn.CrossEntropyLoss()
    epoch = 10

    # 训练模型
    train(model, train_loader, criterion, optimizer, epoch)
    # 保存模型
    print("训练完成")
    torch.save(model, "adapter_with_adapter.pth")
    print("模型已保存")

    # 测试模型
    model = torch.load("adapter_with_adapter.pth")
    test(model, test_loader)
    print("总耗时:{}min".format((time.time() - start_time) / 60))
