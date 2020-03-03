# -*- coding: utf-8 -*-

"""
@date: 2020/3/2 上午8:07
@file: detector.py
@author: zj
@description: 车辆类别检测器
"""

import os
import copy
import cv2
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import selectivesearch

from utils.util import parse_car_csv

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # 加载CNN模型
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('./models/linear_svm_alexnet_car.pth'))
    model.eval()
    # print(model)
    model = model.to(device)
    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    # 创建selectivesearch对象
    gs = selectivesearch.get_selective_search()

    car_root_dir = './data/voc_car/'
    val_root_dir = os.path.join(car_root_dir, 'val')
    samples = parse_car_csv(val_root_dir)

    for sample_name in samples:
        jpeg_path = os.path.join(val_root_dir, 'JPEGImages', sample_name + ".jpg")
        annotation_path = os.path.join(val_root_dir, 'Annotations', sample_name + ".xml")

        img = cv2.imread(jpeg_path)
        dst = copy.deepcopy(img)

        # 候选区域建议
        selectivesearch.config(gs, img, strategy='f')
        rects = selectivesearch.get_rects(gs)
        print('候选区域建议数目： %d' % len(rects))

        rects_transform = transform(rects)
        print(rects_transform.shape)
        exit(0)

        for rect in rects:
            xmin, ymin, xmax, ymax = rect
            rect_img = img[ymin:ymax, xmin:xmax]

            rect_transform = transform(rect_img).to(device)
            output = model(rect_transform.unsqueeze(0))[0]

            if torch.argmax(output).item() == 1:
                """
                预测为汽车
                """
                cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
                print(rect, output)

        cv2.imshow('img', dst)
        cv2.waitKey(0)
