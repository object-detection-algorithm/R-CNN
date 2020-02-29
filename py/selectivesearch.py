# -*- coding: utf-8 -*-

"""
@author: zj
@file:   selectivesearch.py
@time:   2020-02-25
"""

import sys
import cv2


def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy()
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


if __name__ == '__main__':
    """
    选择性搜索算法操作
    """
    gs = get_selective_search()

    img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print(rects)
