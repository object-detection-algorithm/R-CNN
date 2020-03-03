# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午7:22
@file: dataset_finetune.py
@author: zj
@description: 创建微调数据集
"""

import time
import numpy as np
import cv2
import os
import xmltodict
import selectivesearch
from utils.util import check_dir
from utils.util import parse_car_csv
from utils.util import parse_xml
from utils.util import iou
from utils.util import compute_ious

car_root_dir = '../../data/voc_car/'
finetune_root_dir = '../../data/finetune_car/'


def parse_annotation_jpeg(samples, annotation_dir, jpeg_dir, dst_root_dir, gs):
    """
    获取正负样本（注：忽略属性difficult为True的标注边界框）
    正样本：候选建议与标注边界框IoU大于等于0.5
    负样本：IoU大于0,小于0.5。为了进一步限制负样本数目，其大小必须大于标注框的1/5
    """
    dst_positive_dir = os.path.join(dst_root_dir, '1')
    dst_nevative_dir = os.path.join(dst_root_dir, '0')
    check_dir(dst_positive_dir)
    check_dir(dst_nevative_dir)

    for sample_name in samples:
        since = time.time()

        annotation_path = os.path.join(annotation_dir, sample_name + '.xml')
        jpeg_path = os.path.join(jpeg_dir, sample_name + '.jpg')

        img = cv2.imread(jpeg_path)
        selectivesearch.config(gs, img, strategy='q')

        # 计算候选建议
        rects = selectivesearch.get_rects(gs)
        # 获取标注边界框
        bndboxs = parse_xml(annotation_path)

        maximum_bndbox_size = 0
        for bndbox in bndboxs:
            xmin, ymin, xmax, ymax = bndbox
            bndbox_size = (ymax - ymin) * (xmax - xmin)
            if bndbox_size > maximum_bndbox_size:
                maximum_bndbox_size = bndbox_size

        # 获取候选建议和标注边界框的IoU
        iou_list = compute_ious(rects, bndboxs)
        for i in range(len(iou_list)):
            xmin, ymin, xmax, ymax = rects[i]
            rect_size = (ymax - ymin) * (xmax - xmin)
            rect_img = img[ymin:ymax, xmin:xmax]

            iou_score = iou_list[i]
            if iou_list[i] >= 0.5:
                # 正样本
                dst_positive_path = os.path.join(dst_positive_dir, '%s-%d.png' % (sample_name, i))
                cv2.imwrite(dst_positive_path, rect_img)
            if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
                # 负样本
                dst_negative_path = os.path.join(dst_nevative_dir, '%s-%d.png' % (sample_name, i))
                cv2.imwrite(dst_negative_path, rect_img)
            else:
                pass

        time_elapsed = time.time() - since
        print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    gs = selectivesearch.get_selective_search()

    check_dir(finetune_root_dir)
    for name in ['train', 'val']:
        data_root_dir = os.path.join(car_root_dir, name)
        samples = parse_car_csv(data_root_dir)
        # print(samples)

        data_annotation_dir = os.path.join(data_root_dir, 'Annotations')
        data_jpeg_dir = os.path.join(data_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(finetune_root_dir, name)
        check_dir(dst_root_dir)
        parse_annotation_jpeg(samples, data_annotation_dir, data_jpeg_dir, dst_root_dir, gs)

    print('done')
