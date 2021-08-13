# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/8 12:28
@file: util.py
@brief: 
"""
import os
import time
from pprint import pprint

import cv2

from common.folder_process import folder_for_save


def optimize_morghology(single_frame, K):
    """
    # 借助形态学处理操作消除噪声点、连通对象
    :param single_frame:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, K))
    frame_parsed = cv2.morphologyEx(single_frame, cv2.MORPH_OPEN, kernel, iterations=1)
    frame_parsed = cv2.morphologyEx(frame_parsed, cv2.MORPH_CLOSE, kernel, iterations=3)
    # thresh = cv2.threshold(frame_parsed, 127, 255, cv2.THRESH_BINARY)[1]

    return frame_parsed


def optimize_morghology2(frame, K):
    """
    # 借助形态学处理操作消除噪声点、连通对象
    :param single_frame:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, K))
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=3)

    return frame

def optimize_morghology3(frame, K):
    """
    # 借助形态学处理操作消除噪声点、连通对象
    :param single_frame:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (K, K))
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=3)
    frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel, iterations=3)
    # thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]

    return frame


def optimize_median(single_frame, K):
    """
    中值滤波对图形进行优化
    :param single_frame:
    :return:
    """
    blur = cv2.medianBlur(single_frame, K)  # 中值滤波
    # thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]  # 阈值处理

    return blur


def saveimg(frame, filename, input_path, output_path, sub_rootpath=None):
    """
    Args:
        frame: 1. 正常帧， 2.字典形式的帧
            {
                mask: 帧  # []
                masks：多个帧的集合  # [[], [], []]
            }
        filename:
        input_path:
        output_path:
        sub_rootpath: 子根目录。同一算法下记录不同操作的检测结果
    Returns:

    """
    if type(frame) == dict:
        output_path_mask = output_path + '_mask'
        output_path_masks = output_path + '_masks'
        save_dir_path_mask = folder_for_save(input_path, output_path_mask, sub_rootpath)
        save_dir_path_masks = folder_for_save(input_path, output_path_masks, sub_rootpath)
        save_dir_path_masks = os.path.join(save_dir_path_masks, filename[:-4])
        if not os.path.exists(save_dir_path_masks):
            os.mkdir(save_dir_path_masks)

        save_path_mask = os.path.join(save_dir_path_mask, filename)
        cv2.imwrite(save_path_mask, frame["mask"])

        for id_m, f in enumerate(frame["masks"]):
            filename = str(id_m) + '.jpg'
            save_path_masks = os.path.join(save_dir_path_masks, filename)
            cv2.imwrite(save_path_masks, f)
    else:
        save_dir_path = folder_for_save(input_path, output_path, sub_rootpath)
        save_path = os.path.join(save_dir_path, filename)
        # print(save_path)
        cv2.imwrite(save_path, frame)


def save_fps(path, name, fps):
    """
    保存fps
    :param path:
    :param name:
    :param fps:
    :return:
    """
    with open(path, 'a+') as f:
        f.write('{}, {:.2f} \n'.format(name, fps))


if __name__ == '__main__':
    pass


