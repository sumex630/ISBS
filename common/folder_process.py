# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/5/5 13:05
@file: folder_process.py
@brief: 
"""
import os
import shutil


def get_input_path(dataset_rootpath):
    """
    文件操作，获取输入数据的路径
    :param dataset_rootpath:
    :return:
    """
    input_list = []
    for category in get_directories(dataset_rootpath):
        category_path = os.path.join(dataset_rootpath, category, category)

        for video in get_directories(category_path):
            video_path = os.path.join(category_path, video, 'input')

            input_list.append(video_path)

    return input_list


def folder_for_save(input_path, output_rootpath, sub_rootpath=None):
    """
    根据输入图片的路径，保存结果图片到指定文件夹内
    :param sub_rootpath:
    :param output_rootpath:'../results/mb_K=200_t=t0.2_'
    :param input_path:../../dataset/baseline/baseline/{}/input'.format(video)
    :return: ../../results/gmm/gmm/baseline/{}/bin%06d.jpg
    """
    input_path_list = input_path.replace('\\', '/').split('/')
    output_rootpath_list = output_rootpath.replace('\\', '/').split('/')
    # 20210812 添加无 input 时的路径处理
    if input_path_list[-1] == "input":
        path_list = input_path_list[-4:-1]  # 去掉input之后 及 dataset及之前
    else:
        # 'data\\bs\\baseline\\office'
        path_list = input_path_list[-3:]
    # 有 sub_rootpath 则拼接，没有则创建一个，以免后期在 output_rootpath
    output_rootpath = os.path.join(output_rootpath, sub_rootpath) if sub_rootpath else os.path.join(output_rootpath, output_rootpath_list[-1])
    path_list[0] = output_rootpath
    output_path = list2path(path_list)  # 输出路径
    # output_path_ = path_is_exist(output_path)  # 判断路径是否存在，不存在则创建
    if not os.path.exists(output_path):
        # os.mkdirs(output_path)
        os.makedirs(output_path)

    return output_path


def dir_is_null(path):
    """
    文件夹下是否为空，不为空删除文件
    :param path:
    :return:
    """
    flag = os.listdir(path)
    if flag:
        # 强制删除文件夹
        shutil.rmtree(path)
        path_is_exist(path)


def path_is_exist(path):
    """
    判断路径是否存在，不存在则创建
    :param path:
    :return:
    """
    output_path_ = ''
    path_list = path.replace('\\', '/').split('/')
    for path_name in path_list:  # 查看输出路径是否存在
        output_path_ = os.path.join(output_path_, path_name)
        if '.\\' not in path_name:
            if not os.path.exists(output_path_):
                os.mkdir(output_path_)

    return output_path_


def list2path(path_list):
    """
    根据list返回 path
    :param path_list:
    :return:
    """
    return os.path.join(*path_list) if path_list else ''


def get_directories(path):
    """
    Return a list of directories name on the specifed path
    :param path:
    :return:
    """
    return [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]


if __name__ == '__main__':
    PATH_ROOT = 'F:/Dataset/CDNet2012/baseline\\baseline\\highway\\input'
    # start_time = time.time()
    # create_dir_is_exist(PATH_ROOT)

    p = folder_for_save(PATH_ROOT, '../../results/gmm_', 'medain')

    print(p)







