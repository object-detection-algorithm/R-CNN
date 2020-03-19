# -*- coding: utf-8 -*-

"""
@date: 2020/3/3 下午7:38
@file: custom_batch_sampler.py
@author: zj
@description: 自定义采样器
"""

import numpy  as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data.custom_finetune_dataset import CustomFinetuneDataset


class CustomBatchSampler(Sampler):

    def __init__(self, num_positive, num_negative, batch_positive, batch_negative) -> None:
        """
        2分类数据集
        每次批量处理，其中batch_positive个正样本，batch_negative个负样本
        @param num_positive: 正样本数目
        @param num_negative: 负样本数目
        @param batch_positive: 单次正样本数
        @param batch_negative: 单次负样本数
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative

        length = num_positive + num_negative
        self.idx_list = list(range(length))

        self.batch = batch_negative + batch_positive
        self.num_iter = length // self.batch

    def __iter__(self):
        sampler_list = list()
        for i in range(self.num_iter):
            tmp = np.concatenate(
                (random.sample(self.idx_list[:self.num_positive], self.batch_positive),
                 random.sample(self.idx_list[self.num_positive:], self.batch_negative))
            )
            random.shuffle(tmp)
            sampler_list.extend(tmp)
        return iter(sampler_list)

    def __len__(self) -> int:
        return self.num_iter * self.batch

    def get_num_batch(self) -> int:
        return self.num_iter


def test():
    root_dir = '../../data/finetune_car/train'
    train_data_set = CustomFinetuneDataset(root_dir)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96)

    print('sampler len: %d' % train_sampler.__len__())
    print('sampler batch num: %d' % train_sampler.get_num_batch())

    first_idx_list = list(train_sampler.__iter__())[:128]
    print(first_idx_list)
    # 单次批量中正样本个数
    print('positive batch: %d' % np.sum(np.array(first_idx_list) < 66517))


def test2():
    root_dir = '../../data/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    train_sampler = CustomBatchSampler(train_data_set.get_positive_num(), train_data_set.get_negative_num(), 32, 96)
    data_loader = DataLoader(train_data_set, batch_size=128, sampler=train_sampler, num_workers=8, drop_last=True)

    inputs, targets = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    test()
