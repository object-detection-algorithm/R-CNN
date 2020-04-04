# -*- coding: utf-8 -*-

"""
@date: 2020/4/3 下午6:55
@file: bbox_regression.py
@author: zj
@description: 边界框回归训练
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import AlexNet

from utils.data.custom_bbox_regression_dataset import BBoxRegressionDataset
import utils.util as util


def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_set = BBoxRegressionDataset(data_root_dir, transform=transform)
    data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)

    return data_loader


def train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    model.train()  # Set model to training mode
    loss_list = list()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        # Iterate over data.
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)

            features = feature_model.features(inputs)
            features = torch.flatten(features, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(features)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            lr_scheduler.step()

        epoch_loss = running_loss / data_loader.dataset.__len__()
        loss_list.append(epoch_loss)

        print('{} Loss: {:.4f}'.format(epoch, epoch_loss))

        # 每训练一轮就保存
        util.save_model(model, './models/bbox_regression_%d.pth' % epoch)

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return loss_list


def get_model(device=None):
    # 加载CNN模型
    model = AlexNet(num_classes=2)
    model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_car.pth'))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


if __name__ == '__main__':
    data_loader = load_data('./data/bbox_regression')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_model = get_model(device)

    # AlexNet最后一个池化层计算得到256*6*6输出
    in_features = 256 * 6 * 6
    out_features = 4
    model = nn.Linear(in_features, out_features)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    loss_list = train_model(data_loader, feature_model, model, criterion, optimizer, lr_scheduler, device=device,
                            num_epochs=12)
    util.plot_loss(loss_list)
