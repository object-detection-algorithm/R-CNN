# -*- coding: utf-8 -*-

"""
@author: zj
@file:   finetune_dataset.py
@time:   2020-02-25
@description: 创建微调CNN模型的数据集
"""

import os
import cv2
import xmltodict
import numpy  as np
import selectivesearch


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_car_csv(csv_path):
    samples = np.loadtxt(csv_path, dtype=np.str)
    return samples


def parse_annotation(annotation_path):
    """
    解析xml文件，返回边界框坐标
    """
    with open(annotation_path, 'rb') as f:
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


def iou(pred_box, target_box):
    xA = np.maximum(target_box[0], pred_box[0])
    yA = np.maximum(target_box[1], pred_box[1])
    xB = np.minimum(target_box[2], pred_box[2])
    yB = np.minimum(target_box[3], pred_box[3])
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    boxBArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

    iou = intersection / (boxAArea + boxBArea - intersection)
    return iou


def compute_ious(rect, bndboxs):
    scores = list()
    for bndbox in bndboxs:
        score = iou(rect, bndbox)
        scores.append(score)
    return np.array(scores)


def create_samples(img, bndboxs):
    gs = selectivesearch.get_selective_search()
    selectivesearch.pretreat(gs, img, strategy='q')
    rects = selectivesearch.get_rects(gs)

    positives = list()
    negatives = list()
    for rect in rects:
        ious = compute_ious(rect, bndboxs)
        if np.max(ious) >= 0.5:
            # 重叠率大于0.5
            positives.append(rect)
        else:
            # 重叠率小于0.5
            negatives.append(rect)

    return positives, negatives


def save_samples(data_dir, rects, sample_name, img):
    """
    保存正负样本到指定文件夹
    """
    num = 0
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        sample = img[ymin:ymax, xmin:xmax]
        sample_path = os.path.join(data_dir, "%s-%d.png" % (sample_name, num))

        cv2.imwrite(sample_path, sample)
        num += 1


if __name__ == '__main__':
    data_root = './data/car'
    jpeg_dir = os.path.join(data_root, 'JPEGImages')
    annotation_dir = os.path.join(data_root, 'Annotations')

    train_dir = os.path.join(data_root, 'finetune')
    positive_dir = os.path.join(train_dir, '1')
    negative_dir = os.path.join(train_dir, '0')

    check_dir(train_dir)
    check_dir(positive_dir)
    check_dir(negative_dir)

    csv_path = os.path.join(data_root, 'car.csv')
    samples = parse_car_csv(csv_path)

    for sample_name in samples:
        jpeg_path = os.path.join(jpeg_dir, sample_name + '.jpg')
        annotation_path = os.path.join(annotation_dir, sample_name + '.xml')

        img = cv2.imread(jpeg_path, cv2.IMREAD_COLOR)
        bndboxs = parse_annotation(annotation_path)
        # 创建正样本、负样本
        positives, negatives = create_samples(img, bndboxs)

        save_samples(positive_dir, positives, sample_name, img)
        save_samples(negative_dir, negatives, sample_name, img)

    print('done')
