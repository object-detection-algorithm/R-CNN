# -*- coding: utf-8 -*-

"""
@date: 2020/3/2 上午8:07
@file: car_detector.py
@author: zj
@description: 车辆类别检测器
"""

import copy
import cv2
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torchvision.transforms as transforms
import selectivesearch
from utils.util import parse_xml


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_transform():
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_model(device=None):
    # 加载CNN模型
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    # model.load_state_dict(torch.load('./models/linear_svm_alexnet_car_4.pth'))
    model.load_state_dict(torch.load('./models/best_linear_svm_alexnet_car.pth'))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


def draw_box_with_text(img, rect_list, score_list):
    """
    绘制边框及其分类概率
    :param img:
    :param rect_list:
    :param score_list:
    :return:
    """
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == '__main__':
    device = get_device()
    transform = get_transform()
    model = get_model(device=device)

    # 创建selectivesearch对象
    gs = selectivesearch.get_selective_search()

    test_img_path = './data/voc_car/val/JPEGImages/000007.jpg'
    test_xml_path = './data/voc_car/val/Annotations/000007.xml'

    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    bndboxs = parse_xml(test_xml_path)
    for bndbox in bndboxs:
        xmin, ymin, xmax, ymax = bndbox
        cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)

    # 候选区域建议
    selectivesearch.config(gs, img, strategy='f')
    rects = selectivesearch.get_rects(gs)
    print('候选区域建议数目： %d' % len(rects))

    # softmax = torch.softmax()

    # 保存正样本边界框以及
    score_list = list()
    positive_list = list()
    for rect in rects:
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            """
            预测为汽车
            """
            probs = torch.softmax(output, dim=0).cpu().numpy()

            score_list.append(probs[1])
            positive_list.append(rect)
            # cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
            print(rect, output, probs)

    draw_box_with_text(dst, positive_list, score_list)

    cv2.imshow('img', dst)
    cv2.waitKey(0)
