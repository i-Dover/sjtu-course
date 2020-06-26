#coding=utf-8
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
import copy
import json
import pickle
import numpy as np
import cv2
from util import *

FPS = 25
LENGTH = 768
WINS = [LENGTH * 8]
#LENGTH = 192
#WINS = [LENGTH * 32]
STEP = LENGTH / 4
FRAME_DIR = './media/F/ActivityNet/frames_'+str(FPS)
data = json.load(open('./test.json'))
USE_FLIPPED = False
classes = generate_classes(data)
test_segment = generate_segment('validation', data, classes, FRAME_DIR)

def generate_roi(video, start, end, stride, split):
    tmp = {}
    tmp['flipped'] = False
    tmp['frames'] = np.array([[0, start, end, stride]])
    tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
    tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
    if not os.path.isfile(os.path.join(tmp['bg_name'], 'image_' + str(end-1).zfill(5) + '.jpg')):
        print (os.path.join(tmp['bg_name'], 'image_' + str(end-1).zfill(5) + '.jpg'))
        raise
    return tmp

def generate_roidb(split):
    VIDEO_PATH1 = os.path.join(FRAME_DIR, 'training')
    VIDEO_PATH2 = os.path.join(FRAME_DIR, 'validation')
    video_list1 = set(os.listdir(VIDEO_PATH1))
    video_list2 = set(os.listdir(VIDEO_PATH2))
    roidb = []
    for vid in segment:
        if vid not in video_list1 and vid not in video_list2:
            continue
        else:
            if vid in video_list1:
                length = len(os.listdir(os.path.join(VIDEO_PATH1, vid)))
            if vid in video_list2:
                length = len(os.listdir(os.path.join(VIDEO_PATH2, vid)))
        if (length==0):
            continue
        for win in WINS:
            stride = int(win / LENGTH)
            step = int(stride * STEP)
            # Forward Direction
            for start in range(0, max(1, length - win + 1), step):
                end = min(start + win, length)
                assert end <= length

                # Add data
                tmp = generate_roi(vid, start, end, stride, split)
                roidb.append(tmp)

                if USE_FLIPPED:
                    flipped_tmp = copy.deepcopy(tmp)
                    flipped_tmp['flipped'] = True
                    roidb.append(flipped_tmp)

            # Backward Direction
            for end in range(length, win-1, - step):
                start = end - win
                assert start >= 0

                # Add data
                tmp = generate_roi(vid, start, end, stride, split)
                roidb.append(tmp)

            if USE_FLIPPED:
                flipped_tmp = copy.deepcopy(tmp)
                flipped_tmp['flipped'] = True
                roidb.append(flipped_tmp)

    return roidb
      
val_roidb = generate_roidb('validation')
print (len(val_roidb))
  
print ("Save dictionary")
pickle.dump(val_roidb, open('val_data_{}fps.pkl'.format(FPS),'wb'), pickle.HIGHEST_PROTOCOL)
