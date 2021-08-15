# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/8/11 19:15
@file: isbs.py
"""
import argparse
import os
import time
from functools import reduce

import numpy as np

import cv2

from common.util import saveimg

parser = argparse.ArgumentParser()

"""
python isbs.py 
-is /home/lthpc/sumex/code/20210807_ISBS/yolact/yolact/results/20210812_masks/score_t=0.005_top_k=15 
-bs /home/lthpc/sumex/code/20210807_ISBS/ViBe/results/matchingThreshold=45 
-o results/yolact_vibe 
-opt 0 
-r 0.3
"""

# Input arguments
parser.add_argument("--IS", help="filepath to the Instance segmentation", default=r"E:\01_PyCharm\01_CV\20210714_instance_segmentation\20210716_yolact\results\202108015_yolact_vibe_masks\mt=45_ratio=0.3_yolact_vibe20210815")
parser.add_argument("--BS", help="filepath to the Background subtraction algorithm", default=r"E:\01_PyCharm\01_CV\20210714_instance_segmentation\20210716_yolact\results\202108015_yolact_vibe_mask\mt=45_ratio=0.3_yolact_vibe20210815")
parser.add_argument("--O", help="filepath to the output_path", default="results/is_bs")
parser.add_argument("--SR", help="filepath to the sub_rootpath", default="")
parser.add_argument("--OPT", help="optimized ", default=0, type=int)
parser.add_argument("--R", help="bs / is", default=0.3, type=float)

args = parser.parse_args()


def optimized(frame, K):
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

def img2float32(path):
    img_gray = cv2.imread(path, 0)
    img_float32 = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)[1].astype(np.float32)

    return img_float32


def isbs():
    filerootpath_is = args.IS  # 实例分割模型检测出的单个mask
    filerootpath_bs = args.BS  # 背景减除算法结果
    ratio = args.R  # 比率
    opt = args.OPT  # 是否使用形态学
    localtime = time.strftime("%Y%m%d", time.localtime())
    # localtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_path = args.O
    sub_rootpath = args.SR if args.SR else "{}_".format(localtime) + "ratio={}".format(ratio) + "_opt={}".format(opt)
    ii = 0  # 计数

    for dirpath, dirnames, filenames in os.walk(filerootpath_bs):
        if filenames and "tunnelExit_0_35fps" in dirpath:  #  and 'boats' in dirpath  corridor traffic
            ii += 1
            print("{} dirpath: {}".format(ii, dirpath))  # 'data\\bs\\baseline\\office'
            for filename_bs in filenames:
                # if "000003" not in filename_bs:
                #     continue
                # print(filename_bs)
                # 背景减除算法 mask
                imgpath_bs = os.path.join(dirpath, filename_bs)
                # try:
                bs_mask = img2float32(imgpath_bs)
                # except:
                #     print(imgpath_bs)

                # 实例分割模型 masks
                filepath_is = os.path.join(filerootpath_is + dirpath.split(filerootpath_bs)[-1], filename_bs[:-4])
                masks = np.zeros_like(bs_mask)

                for filename_is in os.listdir(filepath_is):
                    imgpath_is = os.path.join(filepath_is, filename_is)
                    # I_ti
                    is_mask = img2float32(imgpath_is)
                    is_mask_pixel_num = (is_mask.reshape(-1) == 1).sum()

                    # 避免实例分割模型 将一整张图片检测成一个mask
                    if is_mask_pixel_num / (is_mask.shape[0] * is_mask.shape[1]) > 0.7:
                        masks = bs_mask
                        break
                    # 二者取 交
                    bs_and_is_mask = cv2.bitwise_and(is_mask, bs_mask)
                    bs_and_is_mask_pixel_num = (bs_and_is_mask.reshape(-1) == 1).sum()
                    # 实例分割模型结果无前景目标
                    if is_mask_pixel_num == 0:
                        continue
                    # 动态前景 叠加
                    if bs_and_is_mask_pixel_num / is_mask_pixel_num > ratio:
                        masks = cv2.bitwise_or(masks, is_mask)

                    # cv2.imshow("bs", cv2.threshold(bs_mask, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8))
                    # cv2.imshow("is_mask", cv2.threshold(is_mask, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8))
                    # cv2.imshow("masks", cv2.threshold(masks, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8))
                    # cv2.waitKey(100)

                # ########## save
                saveimg(masks, filename_bs, dirpath, output_path, sub_rootpath)


if __name__ == '__main__':
    isbs()

