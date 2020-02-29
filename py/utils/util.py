# -*- coding: utf-8 -*-

"""
@date: 2020/2/29 下午7:31
@file: util.py
@author: zj
@description: 
"""

import os


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
