from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
import pandas as pd


def init_ff(phase, level='frame', n_frames=8):
    dataset_path = 'data/FaceForensics++/original_sequences/youtube/c23/frames/'

    image_list = []
    label_list = []

    folder_list = sorted(glob(dataset_path + '*'))
    filelist = []
    list_dict = json.load(open(f'data/FaceForensics++/{phase}.json', 'r'))
    for i in list_dict:
        filelist += i
    # 只保留出现在 {phase}.json 里的视频
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

    if level == 'video':
        label_list = [0] * len(folder_list)
        return folder_list, label_list

    # level == 'frame'
    for folder in folder_list:
        images_temp = sorted(glob(folder + '/*.png'))

        if len(images_temp) == 0:
            continue

        # 如果帧数多于 n_frames，就随机抽 n_frames 张
        if len(images_temp) > n_frames:
            # 随机挑选 n_frames 个索引（不放回），再排序保持时间顺序
            idx = np.random.choice(len(images_temp), n_frames, replace=False)
            idx = np.sort(idx)
            images_temp = [images_temp[j] for j in idx]

        image_list += images_temp
        label_list += [0] * len(images_temp)

    return image_list, label_list
