# -*- coding: utf-8 -*-

"""
@date: 2020/4/3 下午7:19
@file: create_bbox_regression_data.py
@author: zj
@description: 创建边界框回归数据集
"""

import os
import shutil
import numpy as np
import utils.util as util

# 正样本边界框数目：37222

if __name__ == '__main__':
    """
    从voc_car/train目录中提取标注边界框坐标
    从finetune_car/train目录中提取训练集正样本坐标（IoU>=0.5），进一步提取IoU>0.6的边界框
    数据集保存在bbox_car目录下
    """
    voc_car_train_dir = '../../data/voc_car/train'
    # ground truth
    gt_annotation_dir = os.path.join(voc_car_train_dir, 'Annotations')
    jpeg_dir = os.path.join(voc_car_train_dir, 'JPEGImages')

    classifier_car_train_dir = '../../data/finetune_car/train'
    # positive
    positive_annotation_dir = os.path.join(classifier_car_train_dir, 'Annotations')

    dst_root_dir = '../../data/bbox_regression/'
    dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
    dst_bndbox_dir = os.path.join(dst_root_dir, 'bndboxs')
    dst_positive_dir = os.path.join(dst_root_dir, 'positive')

    util.check_dir(dst_root_dir)
    util.check_dir(dst_jpeg_dir)
    util.check_dir(dst_bndbox_dir)
    util.check_dir(dst_positive_dir)

    samples = util.parse_car_csv(voc_car_train_dir)
    res_samples = list()
    total_positive_num = 0
    for sample_name in samples:
        # 提取正样本边界框坐标（IoU>=0.5）
        positive_annotation_path = os.path.join(positive_annotation_dir, sample_name + '_1.csv')
        positive_bndboxes = np.loadtxt(positive_annotation_path, dtype=np.int, delimiter=' ')
        # 提取标注边界框
        gt_annotation_path = os.path.join(gt_annotation_dir, sample_name + '.xml')
        bndboxs = util.parse_xml(gt_annotation_path)
        # 计算符合条件（IoU>0.6）的候选建议
        positive_list = list()
        if len(positive_bndboxes.shape) == 1 and len(positive_bndboxes) != 0:
            scores = util.iou(positive_bndboxes, bndboxs)
            if np.max(scores) > 0.6:
                positive_list.append(positive_bndboxes)
        elif len(positive_bndboxes.shape) == 2:
            for positive_bndboxe in positive_bndboxes:
                scores = util.iou(positive_bndboxe, bndboxs)
                if np.max(scores) > 0.6:
                    positive_list.append(positive_bndboxe)
        else:
            pass

        # 如果存在正样本边界框（IoU>0.6），那么保存相应的图片以及标注边界框
        if len(positive_list) > 0:
            # 保存图片
            jpeg_path = os.path.join(jpeg_dir, sample_name + ".jpg")
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + ".jpg")
            shutil.copyfile(jpeg_path, dst_jpeg_path)
            # 保存标注边界框
            dst_bndbox_path = os.path.join(dst_bndbox_dir, sample_name + ".csv")
            np.savetxt(dst_bndbox_path, bndboxs, fmt='%s', delimiter=' ')
            # 保存正样本边界框
            dst_positive_path = os.path.join(dst_positive_dir, sample_name + ".csv")
            np.savetxt(dst_positive_path, np.array(positive_list), fmt='%s', delimiter=' ')

            total_positive_num += len(positive_list)
            res_samples.append(sample_name)
            print('save {} done'.format(sample_name))
        else:
            print('-------- {} 不符合条件'.format(sample_name))

    dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
    np.savetxt(dst_csv_path, res_samples, fmt='%s', delimiter=' ')
    print('total positive num: {}'.format(total_positive_num))
    print('done')
