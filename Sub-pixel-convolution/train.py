# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/7/14 16:28
# @Author  : LINYANZHEN
# @File    : train.py
import random

from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR
import os
import model
import time
import super_resolution_utils as utils
import torch
import torch.nn as nn
import classify_utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import test


def test_image(model, test_loader, save_path):
    i = random.randint(0, test_loader.__len__())
    x, lr_cb, lr_cr, y, hr_cb, hr_cr = test_loader.dataset[i]
    x = torch.unsqueeze(x, dim=0)
    x = x.to(device)
    # 控制亮度值，去掉噪点
    out = model(x).clip(0, 1)
    # 降维，因为从dataloader出来的tensor会有batchsize和通道数量这2维度，不方便后面合成图像
    x = x.squeeze()
    out = out.squeeze()
    y = y.squeeze()
    y = utils.tensor_to_image(y)
    out = utils.tensor_to_image(out)
    lr_y = utils.tensor_to_image(x)
    lr_cb = utils.tensor_to_image(lr_cb)
    lr_cr = utils.tensor_to_image(lr_cr)
    hr_cb = utils.tensor_to_image(hr_cb)
    hr_cr = utils.tensor_to_image(hr_cr)
    # 用双三次插值将cbcr通道放大到hr
    out_cb = lr_cb.resize(out.size, Image.BICUBIC)
    out_cr = lr_cr.resize(out.size, Image.BICUBIC)

    # 图片合并输出
    hr = Image.merge('YCbCr', [y, hr_cb, hr_cr]).convert('RGB')
    out_img = Image.merge('YCbCr', [out, out_cb, out_cr]).convert('RGB')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    hr.save(os.path.join(save_path, "hr_y.jpg"))
    out_img.save(os.path.join(save_path, "out.jpg"))
    hr = transforms.ToTensor()(hr)
    out_img = transforms.ToTensor()(out_img)
    psnr = utils.calaculate_psnr(hr, out_img)
    print("test img psnr:{}".format(psnr))
    return psnr


def train_and_val(model, train_loader, val_loader, criterion, optimizer, epoch):
    psnr_list = []
    loss_list = []
    best_psnr = 0
    for i in range(epoch):
        epoch_psnr = 0
        epoch_loss = 0
        count = 0
        epoch_start_time = time.time()
        print("epoch[{}/{}]".format(i, epoch))
        model.train()
        for index, (x, y) in enumerate(train_loader, 0):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            optimizer.zero_grad()
            # print(out.size())
            # print(y.size())
            loss = criterion(out, y)
            loss.backward()
            epoch_loss += loss.item()
            # print(epoch_loss)
            optimizer.step()
            count += len(x)
            # y = y.cpu()
            # out = out.cpu()
        epoch_loss /= count
        count = 0
        model.eval()
        for index, (x, y) in enumerate(val_loader, 0):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            epoch_psnr += utils.calaculate_psnr(y, out)
            count += len(x)
        epoch_psnr /= count

        psnr_list.append(epoch_psnr)
        loss_list.append(epoch_loss)
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            # 保存最优模型
            torch.save(model, "sub_pixel_convolution.pth")
            print("模型已保存")

        print("psnr:{}  best psnr:{}".format(epoch_psnr, best_psnr))
        print("mseloss:{}".format(epoch_loss))
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_list, 'b', label='psnr')
    plt.legend()
    plt.grid()
    plt.title('best psnr=%5.2f' % best_psnr)
    plt.savefig('psnr.jpg', dpi=256)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, 'r', label='loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss.jpg', dpi=256)
    plt.close()


if __name__ == '__main__':
    start_time = time.time()

    device = torch.device('cuda:0')
    train_loader, test_loader = utils.get_super_resolution_dataloader("Aircraft_h5py")
    # cudnn.benchmark = True
    model = model.Sub_pixel_conv(upscale_factor=3)
    model = model.to(device)
    lr = 1e-3
    epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam([
    #     {'params': model.first_part.parameters()},
    #     {'params': model.last_part.parameters(), 'lr': lr * 10}
    # ], lr=lr)
    # 调整学习率，在第40，80个epoch时改变学习率
    scheduler = MultiStepLR(optimizer, milestones=[int(epoch * 0.4), int(epoch * 0.8)], gamma=0.1)
    criterion = nn.MSELoss()
    # 训练模型
    # train(model, train_loader, criterion, optimizer, epoch)
    train_and_val(model, train_loader, test_loader, criterion, optimizer, epoch)
    # 保存模型
    print("训练完成")

    # 测试模型
    model = torch.load("sub_pixel_convolution.pth")
    train_loader, test_loader = utils.get_super_resolution_dataloader("Aircraft")
    # 测试图片保存文件夹
    save_path = "test"
    test_image(model, test_loader, save_path)
    print("总耗时:{}min".format((time.time() - start_time) / 60))
