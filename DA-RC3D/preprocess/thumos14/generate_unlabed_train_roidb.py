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
import subprocess
import numpy as np
import cv2
from da_util import *
import glob

FPS = 25
ext = '.mp4'
LENGTH = 768
min_length = 3
overlap_thresh = 0.7
STEP = LENGTH / 4
WINS = [LENGTH * 1]
FRAME_DIR = 'media/F/THUMOS14'
META_DIR = os.path.join(FRAME_DIR, 'annotation_')


def generate_roi(rois, video, start, end, stride, split):
  tmp = {}
  tmp['wins'] = ( rois[:,:2] - start ) / stride
  tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]
  tmp['gt_classes'] = rois[:,2]
  tmp['max_classes'] = rois[:,2]
  tmp['max_overlaps'] = np.ones(len(rois))
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
  tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
  if not os.path.isfile(os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg')):
    print (os.path.join(FRAME_DIR, split, video, 'image_' + str(end-1).zfill(5) + '.jpg'))
    raise
  return tmp

def generate_roidb(split, segment):
  VIDEO_PATH = os.path.join(FRAME_DIR, split)
  video_list = set(os.listdir(VIDEO_PATH))
  duration = []
  roidb = []
  for vid in segment:
    if vid in video_list:
      length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))
      db = np.array(segment[vid])
      if len(db) == 0:
        continue
      db[:,:2] = db[:,:2] * FPS

      for win in WINS:
        # inner of windows
        stride = int(win / LENGTH)
        # Outer of windows
        step = int(stride * STEP)
        # Forward Direction
        for start in range(0, max(1, length - win + 1), step):
          end = min(start + win, length)
          assert end <= length
          rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

          # Remove duration less than min_length
          if len(rois) > 0:
            duration = rois[:,1] - rois[:,0]
            rois = rois[duration >= min_length]

          # Remove overlap less than overlap_thresh
          if len(rois) > 0:
            time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
            overlap = time_in_wins / (rois[:,1] - rois[:,0])
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois = rois[overlap >= overlap_thresh]

          # Append data
          if len(rois) > 0:
            rois[:,0] = np.maximum(start, rois[:,0])
            rois[:,1] = np.minimum(end, rois[:,1])
            tmp = generate_roi(rois, vid, start, end, stride, split)
            roidb.append(tmp)
            if USE_FLIPPED:
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)

        # Backward Direction
        for end in range(length, win-1, - step):
          start = end - win
          assert start >= 0
          rois = db[np.logical_not(np.logical_or(db[:,0] >= end, db[:,1] <= start))]

          # Remove duration less than min_length
          if len(rois) > 0:
            duration = rois[:,1] - rois[:,0]
            rois = rois[duration > min_length]

          # Remove overlap less than overlap_thresh
          if len(rois) > 0:
            time_in_wins = (np.minimum(end, rois[:,1]) - np.maximum(start, rois[:,0]))*1.0
            overlap = time_in_wins / (rois[:,1] - rois[:,0])
            assert min(overlap) >= 0
            assert max(overlap) <= 1
            rois = rois[overlap > overlap_thresh]

          # Append data
          if len(rois) > 0:
            rois[:,0] = np.maximum(start, rois[:,0])
            rois[:,1] = np.minimum(end, rois[:,1])
            tmp = generate_roi(rois, vid, start, end, stride, split)
            roidb.append(tmp)
            if USE_FLIPPED:
               flipped_tmp = copy.deepcopy(tmp)
               flipped_tmp['flipped'] = True
               roidb.append(flipped_tmp)

  return roidb

def get_segment(dir):
  segment={}
  with open(os.path.join(dir), 'r') as f:
    lines = f.readlines()
    for l in lines:
      if 'video' in l:
        vid_name = l.strip().split()[0]
        segment[vid_name] = []
      if '[[' in l:
        l = l.strip()[1:-2]
        l = l.split('],')
        for i in range(len(l)):
          tmp = l[i].strip()[1:]
          tmp = tmp.split(', ')
          start = float(tmp[0])
          end = float(tmp[1])
          label = float(tmp[2])
          segment[vid_name].append([start, end, label])

  return segment

def union(segment1,segment2):
  for vid in segment1:
    if vid in segment2:
      for i in range(len(segment2[vid])):
        segment1[vid].append(segment2[vid][i])
  for vid in segment1:
    segment1[vid].sort(key=lambda x: x[0])
  return segment1

def t_generate_roidb(split, segment, limit):
  VIDEO_PATH = os.path.join(FRAME_DIR, split)
  video_list = set(os.listdir(VIDEO_PATH))
  duration = []
  roidb = []
  i = 0
  for vid in segment:
    if vid in video_list:
      length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))

      for win in WINS:
        # inner of windows
        stride = int(win / LENGTH)
        # Outer of windows
        step = int(stride * STEP)-40
        # Forward Direction
        for start in range(0, max(1, length - win + 1), step):
          i = i+2
          if i <= limit:
            end = min(start + win, length)
            assert end <= length
            rois = np.array(segment[vid])
            rois[:,0]=start
            rois[:,1]=end
            # Append data
            if len(rois) > 0:
              tmp = generate_roi(rois, vid, start, end, stride, split)
              roidb.append(tmp)
              if USE_FLIPPED:
                flipped_tmp = copy.deepcopy(tmp)
                flipped_tmp['flipped'] = True
                roidb.append(flipped_tmp)
        # Backward Direction
        for end in range(length, win - 1, - step):
          i = i+2
          if i <= limit:
            start = end - win
            assert start >= 0
            rois = np.array(segment[vid])
            rois[:, 0] = start
            rois[:, 1] = end
            # Append data
            if len(rois) > 0:
              tmp = generate_roi(rois, vid, start, end, stride, split)
              roidb.append(tmp)
              if USE_FLIPPED:
                flipped_tmp = copy.deepcopy(tmp)
                flipped_tmp['flipped'] = True
                roidb.append(flipped_tmp)

  return roidb



if __name__ == '__main__':

    USE_FLIPPED = True
    print('Generate Source Training unlabeled Segments')
    s_l_train_segment = get_segment('clust_label.txt')
    s_train_segment = dataset_label_parser(META_DIR + 'val', 'val', is_target=False, use_ambiguous=False)
    segment=union(s_train_segment,s_l_train_segment)
    s_train_roidb = generate_roidb('val',segment)
    print (len(s_train_roidb))
    print ("Save dictionary")
    pickle.dump(s_train_roidb, open('train_data_25fps_flipped_s_label.pkl','wb'), pickle.HIGHEST_PROTOCOL)

    print('Generate Target Training Segments')
    t_train_segment = dataset_label_parser(META_DIR + 'val', 'val', is_target=True, use_ambiguous=False)
    t_train_roidb = t_generate_roidb('val', t_train_segment, len(s_train_roidb))
    print(len(t_train_roidb))
    print("Save dictionary")
    pickle.dump(t_train_roidb, open('train_data_25fps_flipped_t.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)






