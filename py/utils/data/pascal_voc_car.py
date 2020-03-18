# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午2:43
@file: pascal_voc_car.py
@author: zj
@description: 从PASCAL VOC 2007数据集中抽取类别Car。保留1/10的数目
"""

import os
import shutil
import random
import numpy as np
import xmltodict
from utils.util import check_dir

suffix_xml = '.xml'
suffix_jpeg = '.jpg'

car_train_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/car_train.txt'
car_val_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/car_val.txt'

voc_annotation_dir = '../../data/VOCdevkit/VOC2007/Annotations/'
voc_jpeg_dir = '../../data/VOCdevkit/VOC2007/JPEGImages/'

car_root_dir = '../../data/voc_car/'


def parse_train_val(data_path):
    """
    提取指定类别图像
    """
    samples = []

    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            res = line.strip().split(' ')
            if len(res) == 3 and int(res[2]) == 1:
                samples.append(res[0])

    return np.array(samples)


def sample_train_val(samples):
    """
    随机采样样本，减少数据集个数（留下1/10）
    """
    for name in ['train', 'val']:
        dataset = samples[name]
        length = len(dataset)

        random_samples = random.sample(range(length), int(length / 10))
        # print(random_samples)
        new_dataset = dataset[random_samples]
        samples[name] = new_dataset

    return samples


# def parse_car(sample_list):
#     """
#     遍历所有的标注文件，筛选包含car的样本
#     """
#
#     car_samples = list()
#     for sample_name in sample_list:
#         annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
#         with open(annotation_path, 'rb') as f:
#             xml_dict = xmltodict.parse(f)
#             # print(xml_dict)
#
#             bndboxs = list()
#             objects = xml_dict['annotation']['object']
#             if isinstance(objects, list):
#                 for obj in objects:
#                     obj_name = obj['name']
#                     difficult = int(obj['difficult'])
#                     if 'car'.__eq__(obj_name) and difficult != 1:
#                         car_samples.append(sample_name)
#             elif isinstance(objects, dict):
#                 obj_name = objects['name']
#                 difficult = int(objects['difficult'])
#                 if 'car'.__eq__(obj_name) and difficult != 1:
#                     car_samples.append(sample_name)
#             else:
#                 pass
#
#     return car_samples


def save_car(car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    """
    保存类别Car的样本图片和标注文件
    """
    for sample_name in car_samples:
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml)
        shutil.copyfile(src_annotation_path, dst_annotation_path)

        src_jpeg_path = os.path.join(voc_jpeg_dir, sample_name + suffix_jpeg)
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpeg)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    csv_path = os.path.join(data_root_dir, 'car.csv')
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')


if __name__ == '__main__':
    samples = {'train': parse_train_val(car_train_path), 'val': parse_train_val(car_val_path)}
    print(samples)
    # samples = sample_train_val(samples)
    # print(samples)

    check_dir(car_root_dir)
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)
        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')

        check_dir(data_root_dir)
        check_dir(data_annotation_dir)
        check_dir(data_jpeg_dir)
        save_car(samples[name], data_root_dir, data_annotation_dir, data_jpeg_dir)

    print('done')
