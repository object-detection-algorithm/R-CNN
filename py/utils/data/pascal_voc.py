# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午2:51
@file: pascal_voc.py
@author: zj
@description: 加载PASCAL VOC 2007数据集
"""

import cv2
import numpy as np
from torchvision.datasets import VOCDetection

if __name__ == '__main__':
    """
    下载PASCAL VOC数据集
    """
    dataset = VOCDetection('../../data', year='2007', image_set='trainval', download=True)

    img, target = dataset.__getitem__(1000)
    img = np.array(img)

    print(target)
    print(img.shape)

    cv2.imshow('img', img)
    cv2.waitKey(0)
