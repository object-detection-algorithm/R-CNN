# -*- coding: utf-8 -*-

"""
@date: 2020/3/18 下午3:37
@file: custom_hard_negative_mining_dataset.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch.utils.data import Dataset


class CustomHardNegativeMiningDataset(Dataset):

    def __init__(self, negative_list, jpeg_images, transform):
        self.negative_list = negative_list
        self.jpeg_images = jpeg_images
        self.transform = transform

    def __getitem__(self, index: int):
        target = 0

        negative_dict = self.negative_list[index]
        xmin, ymin, xmax, ymax = negative_dict['rect']
        image_id = negative_dict['image_id']

        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        image = self.transform(image)

        return image, target, negative_dict

    def __len__(self) -> int:
        return len(self.negative_list)
