# -*- coding: utf-8 -*-

"""
@date: 2020/3/3 下午7:06
@file: custom_dataset.py
@author: zj
@description: 自定义微调数据类
"""

import numpy  as np
import os
import cv2
from torch.utils.data import Dataset

from utils.util import parse_car_csv


class CustomDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir)
        jpeg_images = [cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg"))
                       for sample_name in samples]

        positive_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_1.csv')
                                for sample_name in samples]
        negative_annotations = [os.path.join(root_dir, 'Annotations', sample_name + '_0.csv')
                                for sample_name in samples]
        positive_sizes = list()
        positive_rects = list()
        negative_sizes = list()
        negative_rects = list()

        for annotation_path in positive_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            positive_rects.extend(rects)
            positive_sizes.append(len(rects))
        for annotation_path in negative_annotations:
            rects = np.loadtxt(annotation_path, dtype=np.int, delimiter=' ')
            negative_rects.extend(rects)
            negative_sizes.append(len(rects))

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.positive_sizes = positive_sizes
        self.positive_rects = positive_rects
        self.negative_sizes = negative_sizes
        self.negative_rects = negative_rects
        self.total_positive_size = np.sum(positive_sizes)
        self.total_negative_size = np.sum(negative_sizes)

    def __getitem__(self, index: int):
        # 定位下标所属图像
        image_id = 0
        if index < self.total_positive_size:
            # 正样本
            target = 1
            xmin, ymin, xmax, ymax = self.positive_rects
            # 寻找所属图像
            for i in range(len(self.positive_sizes) - 1):
                if np.sum(self.positive_sizes[:i]) <= index <= np.sum(self.positive_sizes[:(i + 1)]):
                    image_id = i
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        else:
            # 负样本
            target = 0
            xmin, ymin, xmax, ymax = self.negative_rects
            # 寻找所属图像
            idx = index - self.total_positive_size
            for i in range(len(self.negative_sizes) - 1):
                if np.sum(self.positive_sizes[:i]) <= idx <= np.sum(self.positive_sizes[:(i + 1)]):
                    image_id = i
            image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return self.total_positive_size + self.total_negative_size
