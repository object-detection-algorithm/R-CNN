# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 下午2:38
@file: linear_svm.py
@author: zj
@description: 
"""

import time
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet

from utils.data.custom_classifier_dataset import CustomClassifierDataset
from utils.data.custom_sampler import CustomSampler
from utils.util import check_dir
from utils.util import save_model

model_path = './models/alexnet_car.pth'


def load_data(data_root_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'val']:
        data_dir = os.path.join(data_root_dir, name)

        data_set = CustomClassifierDataset(data_dir, transform=transform)
        sampler = CustomSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)

        data_loader = DataLoader(data_set, batch_size=128, sampler=sampler, num_workers=8, drop_last=True)
        data_loaders[name] = data_loader
        data_sizes[name] = len(sampler)
    return data_loaders, data_sizes


def hinge_loss(outputs, labels):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss


def add_hard_negative(fp_mask, cache_dicts, hard_negative_list, phase='train'):
    fp_rects = cache_dicts['rect'][fp_mask]
    fp_image_ids = cache_dicts['image_id'][fp_mask]
    # fp_image_name = cache_dicts['image_name'][fp_mask]

    for i in range(len(fp_rects)):
        # 创建误认为正样本的负样本
        fp_dict = dict()
        fp_dict['rect'] = fp_rects[i]
        fp_dict['image_id'] = fp_image_ids[i]
        # fp_dict['image_name'] = fp_image_name[i]

        # 如果已存在，那么不添加
        is_exist = False
        for hard_negative_dict in hard_negative_list:
            if (hard_negative_dict['image_id'] == fp_dict['image_id']) and \
                    (torch.sum(hard_negative_dict['rect'] == fp_dict['rect']) == 4):
                is_exist = True
        if not is_exist:
            hard_negative_list.append(fp_dict)

    return hard_negative_list


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 保存训练过程中误识别为正样本的负样本，用于下一次的训练
        hard_negative_dict = dict()
        hard_negative_dict['train'] = list()
        hard_negative_dict['val'] = list()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 输出正负样本数
            data_set = data_loaders[phase].dataset
            print('{} - positive_num: {} - negative_num: {}'.format(
                phase, data_set.get_positive_num(), data_set.get_negative_num()))
            print('data_sizers: ' % (str(data_sizes)))

            # Iterate over data.
            for inputs, labels, cache_dicts in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 假阳性掩码
                fp_mask = (labels == 0) & (preds == 1)
                hard_negative_dict[phase] = add_hard_negative(fp_mask, cache_dicts, hard_negative_dict[phase],
                                                              phase=phase)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

            print('{} - hard negative list: %d' % (phase, len(hard_negative_dict[phase])))
            # 训练完成后，重置负样本，进行hard negatives mining
            data_set = data_loaders[phase].dataset
            data_set.set_negative_list(hard_negative_dict[phase])
            sampler = CustomSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)
            data_loaders[phase] = DataLoader(data_set, batch_size=128, sampler=sampler, num_workers=8, drop_last=True)
            # 重置数据集大小
            data_sizes[phase] = len(sampler)

        # 每训练一轮就保存
        save_model(model, 'models/linear_svm_alexnet_car_%d.pth' % epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    data_loaders, data_sizes = load_data('./data/classifier_car')

    # 加载CNN模型
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 固定特征提取
    for param in model.parameters():
        param.requires_grad = False
    # 创建SVM分类器
    model.classifier[6] = nn.Linear(num_features, num_classes)
    # print(model)
    model = model.to(device)

    criterion = hinge_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=25, device=device)
    # 保存最好的模型参数
    save_model(best_model, 'models/best_linear_svm_alexnet_car.pth')
