# -*- coding: utf-8 -*-

"""
@author: zj
@file:   create_data.py
@time:   2020-02-24
"""

from torchvision.datasets import VOCDetection

if __name__ == '__main__':
    """
    下载PASCAL VOC数据集
    """
    dataset = VOCDetection('./data', year='2007', image_set='trainval', download=True)