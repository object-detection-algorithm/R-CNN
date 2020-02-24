# -*- coding: utf-8 -*-

"""
@author: zj
@file:   create_car_positives.py
@time:   2020-02-24
"""

import xmltodict
import numpy as np
import os


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
                if 'car'.__eq__(obj_name):
                    bndbox = obj['bndbox']
                    bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        elif isinstance(objects, dict):
            obj_name = objects['name']
            if 'car'.__eq__(obj_name):
                bndbox = objects['bndbox']
                bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        else:
            pass

        return bndboxs


def parse_img(img_dir, xml_dir, sample_name):
    xml_path = os.path.join(xml_dir, sample_name + '.xml')
    bndboxs = parse_xml(xml_path)
    print(bndboxs)


if __name__ == '__main__':
    """
    创建汽车类别正样本
    """
    car_root = './data/car'
    xml_dir = os.path.join(car_root, 'Annotations')
    jpeg_dir = os.path.join(car_root, 'JPEGImages')

    car_samples = parse_csv(car_root)
    # print(car_samples)

    for car_sample in car_samples:
        parse_img(jpeg_dir, xml_dir, car_sample)
        exit(0)
