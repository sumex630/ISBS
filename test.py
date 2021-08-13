# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/8/11 20:46
@file: test.py
"""
import cv2
import numpy as np
import time

a = np.array([
    [0, 0, 255, 0],
    [255, 0, 255, 255]
], dtype=np.uint8)

b = np.array([
    [0, 0, 255, 255],
    [0, 0, 255, 255]
], dtype=np.uint8)

c = cv2.bitwise_and(a, b)
d = cv2.bitwise_or(a, b)

e = a + b

print(c)
