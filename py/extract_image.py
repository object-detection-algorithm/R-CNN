# -*- coding: utf-8 -*-

"""
@author: zj
@file:   extract_image.py
@time:   2020-02-24
"""

import os
import shutil
import numpy as np


def parse_trainval():
    """
    提取汽车类别图像
    """
    samples = []

    data_path = './data/VOCdevkit/VOC2007/ImageSets/Main/car_trainval.txt'
    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            res = line.strip().split(' ')
            if len(res) == 3:
                if int(res[2]) == 1:
                    samples.append(res[0])

    return samples


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def save_samples(sample_list):
    sample_root = './data/car'
    check_dir(sample_root)

    annotation_dir = './data/VOCdevkit/VOC2007/Annotations'
    jpeg_dir = './data/VOCdevkit/VOC2007/JPEGImages'

    suffix_xml = '.xml'
    suffix_jpeg = '.jpg'

    for sample_name in sample_list:
        src_annotation_path = os.path.join(annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(sample_root, sample_name + suffix_xml)
        shutil.copyfile(src_annotation_path, dst_annotation_path)

        src_jpeg_path = os.path.join(jpeg_dir, sample_name + suffix_jpeg)
        dst_jpeg_path = os.path.join(sample_root, sample_name + suffix_jpeg)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    print('done')


if __name__ == '__main__':
    car_samples = parse_trainval()
    print(car_samples)

    save_samples(car_samples)
