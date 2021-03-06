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
from pprint import pprint

import numpy as np

import cv2
import torch

from common.stats import ConfusionMatrix, get_temporalROI, get_roi, get_gt, get_stats
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
parser.add_argument("--IS", help="filepath to the Instance segmentation", default="/home/lthpc/sumex/code/20210807_ISBS/yolact/yolact/results/20210812_masks/score_t=0.005_top_k=15")
parser.add_argument("--BS", help="filepath to the Background subtraction algorithm", default="/home/lthpc/sumex/code/20210807_ISBS/ViBe/results/matchingThreshold=45")
parser.add_argument("--DS", help="filepath to the datasets", default=r"/home/lthpc/sumex/datasets/cdnet2014/dataset")
parser.add_argument("--OP", help="filepath to the output_path", default="results/yolact_vibe_ft")
parser.add_argument("--SR", help="filepath to the sub_rootpath", default="")
parser.add_argument("--FT", help="filepath to the fine tuning", default="fine_tuning")
parser.add_argument("--OPT", help="optimized ", default=0, type=int)
parser.add_argument("--R", help="bs / is", default=0.3, type=float)

args = parser.parse_args()

filerootpath_is = args.IS  # 实例分割模型检测出的单个mask
filerootpath_bs = args.BS  # 背景减除算法结果
opt = args.OPT  # 是否使用形态学
localtime = time.strftime("%Y%m%d", time.localtime())
# localtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
datasets_path = args.DS
output_path = args.OP
sub_rootpath = args.SR if args.SR else "{}_".format(localtime) + "_opt={}".format(opt) + "ft"
fine_tuning_path = args.FT


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


def save_ft(path, data):
    if not os.path.exists(path):
        os.mkdir(path)
    ft_path = os.path.join(path, sub_rootpath + ".txt")
    with open(ft_path, "w") as f:
        f.write(data)


def img2float32(path):
    img_gray = cv2.imread(path, 0)
    img_float32 = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)[1].astype(np.float32)

    return img_float32


def avg_ft(data):
    """
    计算所有视频序列的最优比值 平均值
    :param data:
    :return:
    """
    ft_list = []
    for v in data.values:
        ft_list.append(v[-1][0])

    return ft_list


def isbs():

    ii = 0  # 计数
    fine_tuning = {}

    for dirpath, dirnames, filenames in os.walk(filerootpath_bs):
        if filenames:  #  and 'boats' in dirpath  corridor traffic
            ii += 1
            print("{} dirpath: {}".format(ii, dirpath))  # 'data\\bs\\baseline\\office'
            dirpath_list = dirpath.replace('\\', '/').split('/')  # 切割路径
            dict_k = dirpath_list[-2]+ "_" + dirpath_list[-1]
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
            CM_6 = ConfusionMatrix()

            for id_bs, filename_bs in enumerate(filenames):
                # 有效帧范围内计算混淆矩阵
                if start_frame_id-1 <= id_bs <= end_frame_id:
                    # 背景减除算法 mask
                    imgpath_bs = os.path.join(dirpath, filename_bs)
                    bs_mask = img2float32(imgpath_bs)

                    # 实例分割模型 masks
                    filepath_is = os.path.join(filerootpath_is + dirpath.split(filerootpath_bs)[-1], filename_bs[:-4])
                    # single mask list
                    is_masks_list = list(map(lambda filename_is: img2float32(os.path.join(filepath_is, filename_is)), os.listdir(filepath_is)))
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
                    masks_6 = np.zeros_like(bs_mask)  # 模板

                    for is_mask in is_masks_list:

                        is_mask_pixel_num = (is_mask.reshape(-1) == 255).sum()

                        # 避免实例分割模型 将一整张图片检测成一个mask
                        if is_mask_pixel_num / (is_mask.shape[0] * is_mask.shape[1]) > 0.8:
                            masks_1 = bs_mask
                            masks_2 = bs_mask
                            masks_3 = bs_mask
                            masks_4 = bs_mask
                            masks_5 = bs_mask
                            masks_6 = bs_mask
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
                        if bs_and_is_mask_pixel_num / is_mask_pixel_num > 0.6:
                            masks_6 = cv2.bitwise_or(masks_6, is_mask)

                    # 计算混淆矩阵  mask, gt, roi
                    roi = get_roi(video_path)
                    gt = get_gt(video_path, filename_bs)

                    CM_1.evaluate(torch.from_numpy(masks_1), torch.from_numpy(gt), torch.from_numpy(roi))
                    CM_2.evaluate(torch.from_numpy(masks_2), torch.from_numpy(gt), torch.from_numpy(roi))
                    CM_3.evaluate(torch.from_numpy(masks_3), torch.from_numpy(gt), torch.from_numpy(roi))
                    CM_4.evaluate(torch.from_numpy(masks_4), torch.from_numpy(gt), torch.from_numpy(roi))
                    CM_5.evaluate(torch.from_numpy(masks_5), torch.from_numpy(gt), torch.from_numpy(roi))
                    CM_6.evaluate(torch.from_numpy(masks_6), torch.from_numpy(gt), torch.from_numpy(roi))

            # 计算指标
            cm_dict = {}
            cm_dict["1"] = get_stats([CM_1.TP.numpy(), CM_1.FP.numpy(), CM_1.FN.numpy(), CM_1.TN.numpy(), 0])["FMeasure"]
            cm_dict["2"] = get_stats([CM_2.TP.numpy(), CM_2.FP.numpy(), CM_2.FN.numpy(), CM_2.TN.numpy(), 0])["FMeasure"]
            cm_dict["3"] = get_stats([CM_3.TP.numpy(), CM_3.FP.numpy(), CM_3.FN.numpy(), CM_3.TN.numpy(), 0])["FMeasure"]
            cm_dict["4"] = get_stats([CM_4.TP.numpy(), CM_4.FP.numpy(), CM_4.FN.numpy(), CM_4.TN.numpy(), 0])["FMeasure"]
            cm_dict["5"] = get_stats([CM_5.TP.numpy(), CM_5.FP.numpy(), CM_5.FN.numpy(), CM_5.TN.numpy(), 0])["FMeasure"]
            cm_dict["6"] = get_stats([CM_6.TP.numpy(), CM_6.FP.numpy(), CM_6.FN.numpy(), CM_6.TN.numpy(), 0])["FMeasure"]

            # 根据 value 对字典排序  (k, v)
            sorted_cms = sorted(cm_dict.items(), key=lambda kv: (kv[1], kv[0]))
            fine_tuning[dict_k] = sorted_cms

    # 计算微调平均值
    ft_list = avg_ft(fine_tuning)
    fine_tuning["mean"] = (np.mean(ft_list), ft_list)

    # 保存
    save_ft(fine_tuning_path, str(fine_tuning))
    # 重新计算 最优情况

    isbs_ft(fine_tuning)


def isbs_ft(ft_data):
    """
    微调
    :param ft_data:
    :return:
    """
    ii = 0
    for dirpath, dirnames, filenames in os.walk(filerootpath_bs):
        if filenames:  #  and 'boats' in dirpath  corridor traffic
            ii += 1
            dirpath_list = dirpath.replace('\\', '/').split('/')  # 切割路径
            dict_k = dirpath_list[-2] + "_" + dirpath_list[-1]
            if dict_k not in list(ft_data.keys()):
                continue

            print("{} dirpath: {}".format(ii, dirpath))  # 'data\\bs\\baseline\\office'
            ratio = int(ft_data[dict_k][-1][0]) * 0.1

            for filename_bs in filenames:
                # 背景减除算法 mask
                imgpath_bs = os.path.join(dirpath, filename_bs)
                bs_mask = img2float32(imgpath_bs)

                # 实例分割模型 masks
                filepath_is = os.path.join(filerootpath_is + dirpath.split(filerootpath_bs)[-1], filename_bs[:-4])
                masks = np.zeros_like(bs_mask)
                if opt:
                    # 是否优化
                    bs_mask = optimized(bs_mask, 3)

                for filename_is in os.listdir(filepath_is):
                    imgpath_is = os.path.join(filepath_is, filename_is)
                    is_mask = img2float32(imgpath_is)

                    is_mask_pixel_num = (is_mask.reshape(-1) == 1).sum()

                    # 避免实例分割模型 将一整张图片检测成一个mask
                    if is_mask_pixel_num / (is_mask.shape[0] * is_mask.shape[1]) > 0.8:
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

                # ########## save
                saveimg(masks, filename_bs, dirpath, output_path, sub_rootpath)


if __name__ == '__main__':
    isbs()

