# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 下午7:17
@file: dataset_classifier.py
@author: zj
@description: 创建分类器数据集
"""

import random
import numpy as np
import cv2
import os
import xmltodict
import selectivesearch
from utils.util import check_dir

car_root_dir = '../../data/voc_car/'
classifier_root_dir = '../../data/classifier_car/'


def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, 'car.csv')
    samples = np.loadtxt(csv_path, dtype=np.str)
    return samples


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

        return np.array(bndboxs)


def iou(pred_box, target_box):
    """
    计算候选建议和标注边界框的IoU
    :param pred_box: 大小为[4]
    :param target_box: 大小为[N, 4]
    :return: [N]
    """
    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * (target_box[:, 3] - target_box[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores


def compute_ious(rects, bndboxs):
    iou_list = list()
    for rect in rects:
        scores = iou(rect, bndboxs)
        iou_list.append(max(scores))
    return iou_list


def parse_annotation_jpeg(samples, annotation_dir, jpeg_dir, dst_root_dir, gs):
    """
    获取正负样本（注：忽略属性difficult为True的标注边界框）
    正样本：标注边界框
    负样本：候选建议与标注边界框的IoU大于0,小于0.3；其大小大于标注框的1/5；随机挑选图像，保持每张图像的正负样本比在1:20以内
    """
    dst_positive_dir = os.path.join(dst_root_dir, '1')
    dst_nevative_dir = os.path.join(dst_root_dir, '0')
    check_dir(dst_positive_dir)
    check_dir(dst_nevative_dir)

    for sample_name in samples:
        annotation_path = os.path.join(annotation_dir, sample_name + '.xml')
        jpeg_path = os.path.join(jpeg_dir, sample_name + '.jpg')

        img = cv2.imread(jpeg_path)
        selectivesearch.config(gs, img, strategy='q')

        # 计算候选建议
        rects = selectivesearch.get_rects(gs)
        # 获取标注边界框
        bndboxs = parse_xml(annotation_path)

        maximum_bndbox_size = 0
        num_positive = len(bndboxs)
        for i in range(num_positive):
            xmin, ymin, xmax, ymax = bndboxs[i]
            bndbox_img = img[ymin:ymax, xmin:xmax]

            bndbox_size = (ymax - ymin) * (xmax - xmin)
            if bndbox_size > maximum_bndbox_size:
                maximum_bndbox_size = bndbox_size

            # 正样本
            dst_positive_path = os.path.join(dst_positive_dir, '%s-%d.png' % (sample_name, i))
            cv2.imwrite(dst_positive_path, bndbox_img)

        # 获取候选建议和标注边界框的IoU
        iou_list = compute_ious(rects, bndboxs)
        # 计算符合条件的候选建议
        negative_list = list()
        for i in range(len(iou_list)):
            xmin, ymin, xmax, ymax = rects[i]
            rect_size = (ymax - ymin) * (xmax - xmin)
            rect_img = img[ymin:ymax, xmin:xmax]

            iou_score = iou_list[i]
            if 0 < iou_score < 0.3 and rect_size > maximum_bndbox_size / 5.0:
                negative_list.append(rect_img)

        # 随机舍去部分负样本，保证正负样本比在1:20之内
        num_negative = len(negative_list)
        ratio = num_negative * 1.0 / num_positive
        if ratio <= 20:
            # 正负样本比在1:20之内，所以保留所有负样本
            for i in range(num_negative):
                # 负样本
                dst_negative_path = os.path.join(dst_nevative_dir, '%s-%d.png' % (sample_name, i))
                cv2.imwrite(dst_negative_path, negative_list[i])
        else:
            idx_negative = random.sample(range(num_negative), num_positive * 20)
            for i in idx_negative:
                # 负样本
                dst_negative_path = os.path.join(dst_nevative_dir, '%s-%d.png' % (sample_name, i))
                cv2.imwrite(dst_negative_path, negative_list[i])


if __name__ == '__main__':
    gs = selectivesearch.get_selective_search()

    check_dir(classifier_root_dir)
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)
        samples = parse_car_csv(data_root_dir)
        # print(samples)

        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(classifier_root_dir, name)
        check_dir(dst_root_dir)
        parse_annotation_jpeg(samples, data_annotation_dir, data_jpeg_dir, dst_root_dir, gs)

    print('done')
