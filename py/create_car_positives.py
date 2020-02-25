# -*- coding: utf-8 -*-

"""
@author: zj
@file:   create_car_positives.py
@time:   2020-02-24
"""

import xmltodict
import numpy as np
import os
import cv2


def parse_csv(car_root):
    csv_path = os.path.join(car_root, 'car.csv')
    car_samples = np.loadtxt(csv_path, dtype=np.str)

    return car_samples


def parse_xml(xml_path):
    """
    解析xml文件，返回正样本边界框坐标
    """
    # print(xml_path)
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        print(xml_dict)

        bndboxs = list()
        objects = xml_dict['annotation']['object']
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj['name']
                difficult = int(obj['difficult'])
                if 'car'.__eq__(obj_name) and difficult != 1:
                    bndbox = obj['bndbox']
                    bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        elif isinstance(objects, dict):
            obj_name = objects['name']
            difficult = int(objects['difficult'])
            if 'car'.__eq__(obj_name) and difficult != 1:
                bndbox = objects['bndbox']
                bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        else:
            pass

        return bndboxs


def parse_img(img_dir, xml_dir, sample_name):
    """
    解析对应xml文件，裁剪汽车边界框
    """
    xml_path = os.path.join(xml_dir, sample_name + '.xml')
    bndboxs = parse_xml(xml_path)
    print(bndboxs)

    img_path = os.path.join(img_dir, sample_name + '.jpg')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # print(img.shape)

    bndboxs_imgs = list()
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_img = img[ymin:ymax, xmin:xmax]
        bndboxs_imgs.append(bndbox_img)

    return bndboxs_imgs, bndboxs


if __name__ == '__main__':
    """
    创建汽车类别正样本
    """
    # 文件路径操作
    car_root = './data/car'
    xml_dir = os.path.join(car_root, 'Annotations')
    jpeg_dir = os.path.join(car_root, 'JPEGImages')

    car_train = os.path.join(car_root, 'train')
    if not os.path.exists(car_train):
        os.mkdir(car_train)

    # 解析csv文件
    car_samples = parse_csv(car_root)
    # print(car_samples)

    for car_sample in car_samples:
        bndboxs_imgs, bndboxs = parse_img(jpeg_dir, xml_dir, car_sample)

        for i in range(len(bndboxs)):
            bndboxs_img = bndboxs_imgs[i]
            xmin, ymin, xmax, ymax = bndboxs[i]

            img_path = os.path.join(car_train, '%s-%d-%d-%d-%d.png' % (car_sample, xmin, ymin, xmax, ymax))
            cv2.imwrite(img_path, bndboxs_img)
