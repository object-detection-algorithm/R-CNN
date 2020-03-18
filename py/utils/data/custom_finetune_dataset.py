# -*- coding: utf-8 -*-

"""
@date: 2020/3/3 下午7:06
@file: custom_finetune_dataset.py
@author: zj
@description: 自定义微调数据类
"""

import numpy  as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.util import parse_car_csv


class CustomFinetuneDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir)

        jpeg_images = [cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg"))
                       for sample_name in samples]

        positive_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_1.csv')
                                for sample_name in samples]
        negative_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_0.csv')
                                for sample_name in samples]

        # 边界框大小
        positive_sizes = list()
        negative_sizes = list()
        # 边界框坐标
        positive_rects = list()
        negative_rects = list()

        for annotation_path in positive_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            # 存在文件为空或者文件中仅有单行数据
            if len(rects.shape) == 1:
                # 是否为单行
                if rects.shape[0] == 4:
                    positive_rects.append(rects)
                    positive_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                positive_rects.extend(rects)
                positive_sizes.append(len(rects))
        for annotation_path in negative_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            # 和正样本规则一样
            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    negative_rects.append(rects)
                    negative_sizes.append(1)
                else:
                    positive_sizes.append(0)
            else:
                negative_rects.extend(rects)
                negative_sizes.append(len(rects))

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_sizes = positive_sizes
        self.negative_sizes = negative_sizes
        self.positive_rects = positive_rects
        self.negative_rects = negative_rects
        self.total_positive_num = int(np.sum(positive_sizes))
        self.total_negative_num = int(np.sum(negative_sizes))

    def __getitem__(self, index: int):
        # 定位下标所属图像
        image_id = len(self.jpeg_images) - 1
        if index < self.total_positive_num:
            # 正样本
            target = 1
            xmin, ymin, xmax, ymax = self.positive_rects[index]
            # 寻找所属图像
            for i in range(len(self.positive_sizes) - 1):
                if np.sum(self.positive_sizes[:i]) <= index < np.sum(self.positive_sizes[:(i + 1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        else:
            # 负样本
            target = 0
            idx = index - self.total_positive_num
            xmin, ymin, xmax, ymax = self.negative_rects[idx]
            # 寻找所属图像
            for i in range(len(self.negative_sizes) - 1):
                if np.sum(self.negative_sizes[:i]) <= idx < np.sum(self.negative_sizes[:(i + 1)]):
                    image_id = i
                    break
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

        # print('index: %d image_id: %d target: %d image.shape: %s [xmin, ymin, xmax, ymax]: [%d, %d, %d, %d]' %
        #       (index, image_id, target, str(image.shape), xmin, ymin, xmax, ymax))
        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return self.total_positive_num + self.total_negative_num

    def get_positive_num(self) -> int:
        return self.total_positive_num

    def get_negative_num(self) -> int:
        return self.total_negative_num


def test(idx):
    root_dir = '../../data/finetune_car/train'
    train_data_set = CustomFinetuneDataset(root_dir)

    print('positive num: %d' % train_data_set.get_positive_num())
    print('negative num: %d' % train_data_set.get_negative_num())
    print('total num: %d' % train_data_set.__len__())

    # 测试id=3/66516/66517/530856
    image, target = train_data_set.__getitem__(idx)
    print('target: %d' % target)

    image = Image.fromarray(image)
    print(image)
    print(type(image))

    # cv2.imshow('image', image)
    # cv2.waitKey(0)


def test2():
    root_dir = '../../data/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    image, target = train_data_set.__getitem__(530856)
    print('target: %d' % target)
    print('image.shape: ' + str(image.shape))


def test3():
    root_dir = '../../data/finetune_car/train'
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data_set = CustomFinetuneDataset(root_dir, transform=transform)
    data_loader = DataLoader(train_data_set, batch_size=128, num_workers=8, drop_last=True)

    inputs, targets = next(data_loader.__iter__())
    print(targets)
    print(inputs.shape)


if __name__ == '__main__':
    # test(159622)
    # test(4051)
    test(24768)
