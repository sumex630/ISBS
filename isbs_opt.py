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

from common.stats import ConfusionMatrix, get_temporalROI, get_roi, get_gt
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
parser.add_argument("-is", "--IS", help="filepath to the Instance segmentation", default="datasets/is")
parser.add_argument("-bs", "--BS", help="filepath to the Background subtraction algorithm", default="datasets/bs")
parser.add_argument("-ds", "--datasets", help="filepath to the datasets", default=r"E:\00_Datasets\dataset2014\dataset")
parser.add_argument("-o",  "--output_path", help="filepath to the output_path", default="results/is_bs")
parser.add_argument("-s",  "--sub_rootpath", help="filepath to the sub_rootpath", default="")
parser.add_argument("-opt", "--optimized", help="optimized ", default=0, type=int)
parser.add_argument("-r", "--ratio", help="bs / is", default=0.3, type=float)

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


def isbs():
    filerootpath_is = args.IS  # 实例分割模型检测出的单个mask
    filerootpath_bs = args.BS  # 背景减除算法结果
    ratio = args.ratio  # 比率
    opt = args.optimized  # 是否使用形态学
    localtime = time.strftime("%Y%m%d", time.localtime())
    # localtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    datasets_path = args.datasets
    output_path = args.output_path
    sub_rootpath = args.sub_rootpath if args.sub_rootpath else "{}_".format(localtime) + "ratio={}".format(ratio) + "_opt={}".format(opt)
    ii = 0  # 计数

    for dirpath, dirnames, filenames in os.walk(filerootpath_bs):
        if filenames:  #  and 'boats' in dirpath  corridor traffic
            ii += 1
            print("{} dirpath: {}".format(ii, dirpath))  # 'data\\bs\\baseline\\office'
            dirpath_list = dirpath.replace('\\', '/').split('/')  # 切割路径
            video_path = os.path.join(datasets_path, dirpath_list[-2], dirpath_list[-1])

            # 有效帧
            vaild_frames = get_temporalROI(video_path)  # 有效帧范围
            start_frame_id = int(vaild_frames[0])  # 起始帧号
            end_frame_id = int(vaild_frames[1])  # 结束帧号


            # 创建 混淆矩阵
            CM_1 = ConfusionMatrix()
            CM_2 = ConfusionMatrix()
            CM_3 = ConfusionMatrix()
            CM_4 = ConfusionMatrix()
            CM_5 = ConfusionMatrix()

            for id_bs, filename_bs in enumerate(filenames):
                # 有效帧范围内计算混淆矩阵
                if start_frame_id-1 <= id_bs <= end_frame_id:
                    # 背景减除算法 mask
                    imgpath_bs = os.path.join(dirpath, filename_bs)
                    bs_mask = cv2.imread(imgpath_bs, 0)

                    # 实例分割模型 masks
                    filepath_is = os.path.join(filerootpath_is + dirpath.split(filerootpath_bs)[-1], filename_bs[:-4])
                    # single mask list
                    is_masks_list = list(map(lambda filename_is: cv2.imread(os.path.join(filepath_is, filename_is), 0), os.listdir(filepath_is)))
                    # 整张实例分割结果
                    is_masks = np.array([reduce(lambda x, y: cv2.bitwise_or(x, y), is_masks_list)])
                    is_masks = np.squeeze(is_masks, 0)

                    # is and bs 融合，获得目标区域的 bs
                    bs_mask = cv2.bitwise_and(is_masks, bs_mask)
                    if opt:
                        # 是否优化
                        bs_mask = optimized(bs_mask, 3)

                    masks_1 = np.zeros_like(bs_mask)  # 模板
                    masks_2 = np.zeros_like(bs_mask)  # 模板
                    masks_3 = np.zeros_like(bs_mask)  # 模板
                    masks_4 = np.zeros_like(bs_mask)  # 模板
                    masks_5 = np.zeros_like(bs_mask)  # 模板

                    for is_mask in is_masks_list:

                        is_mask_pixel_num = (is_mask.reshape(-1) == 255).sum()

                        # 避免实例分割模型 将一整张图片检测成一个mask
                        if is_mask_pixel_num / (is_mask.shape[0] * is_mask.shape[1]) > 0.8:
                            masks_1 = bs_mask
                            masks_2 = bs_mask
                            masks_3 = bs_mask
                            masks_4 = bs_mask
                            masks_5 = bs_mask
                            break

                        # 二者取 交
                        bs_and_is_mask = cv2.bitwise_and(is_mask, bs_mask)
                        bs_and_is_mask_pixel_num = (bs_and_is_mask.reshape(-1) == 255).sum()
                        # 实例分割模型结果无前景目标
                        if is_mask_pixel_num == 0:
                            continue

                        # 动态前景 叠加
                        if bs_and_is_mask_pixel_num / is_mask_pixel_num > 0.1:
                            masks_1 = cv2.bitwise_or(masks_1, is_mask)
                        if bs_and_is_mask_pixel_num / is_mask_pixel_num > 0.2:
                            masks_2 = cv2.bitwise_or(masks_2, is_mask)
                        if bs_and_is_mask_pixel_num / is_mask_pixel_num > 0.3:
                            masks_3 = cv2.bitwise_or(masks_3, is_mask)
                        if bs_and_is_mask_pixel_num / is_mask_pixel_num > 0.4:
                            masks_4 = cv2.bitwise_or(masks_4, is_mask)
                        if bs_and_is_mask_pixel_num / is_mask_pixel_num > 0.5:
                            masks_5 = cv2.bitwise_or(masks_5, is_mask)

                    # 计算混淆矩阵  mask, gt, roi
                    roi = get_roi(video_path)
                    gt = get_gt(video_path, filename_bs)

                    CM_1.evaluate(masks_1, gt, roi)
                    CM_2.evaluate(masks_2, gt, roi)
                    CM_3.evaluate(masks_3, gt, roi)
                    CM_4.evaluate(masks_4, gt, roi)
                    CM_5.evaluate(masks_5, gt, roi)



                    # ########## save
                    # saveimg(masks, filename_bs, dirpath, output_path, sub_rootpath)


if __name__ == '__main__':
    isbs()

