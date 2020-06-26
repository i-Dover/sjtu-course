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
from da_util import *
import subprocess
import shutil
import errno
import glob
from collections import defaultdict
import math

FPS = 25
ext = '.mp4'
LENGTH = 16
STEP = LENGTH
WINS = [LENGTH * 1]
FRAME_DIR = 'media/F/THUMOS14'
META_DIR = os.path.join(FRAME_DIR, 'annotation_')


def generate_roi(rois, video, split):
  tmp = {}
  tmp['wins'] = np.array([[rois[2], rois[3]]])
  tmp['durations'] = np.array([16.])
  tmp['gt_classes'] = np.array([rois[4]])
  tmp['max_classes'] = np.array([1.])
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, rois[0], rois[1], 1]])
  tmp['bg_name'] = os.path.join(FRAME_DIR, split, video)
  tmp['fg_name'] = os.path.join(FRAME_DIR, split, video)
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

      rois = db[np.logical_and(db[:, 1] - db[:, 0] >= 16, db[:, 0] + 16 <= length)]
      for s in rois:
        start = int(s[0])
        end = int(s[1])
        dbraw = s[:2] / FPS
        for p in range(start, min(end, length) - 16, 16):
          frame=[p, p + 16, dbraw[0], dbraw[1], 1]
          tmp = generate_roi(frame, vid,split)
          roidb.append(tmp)


  return roidb

def dataset_label(meta_dir, split):
  class_id = defaultdict(int)
  with open(os.path.join(meta_dir, 'detclasslist.txt'), 'r') as f:
    lines = f.readlines()
    for l in lines:
      cname = l.strip().split()[-1]
      cid = int(l.strip().split()[0])
      class_id[cname] = cid

    segment = {}

  for cname in class_id.keys():
    tmp = '{}_{}.txt'.format(cname, split)
    with open(os.path.join(meta_dir, tmp)) as f:
      lines = f.readlines()
      for l in lines:
        vid_name = l.strip().split()[0]
        start_t = float(l.strip().split()[1])
        end_t = float(l.strip().split()[2])
        #video_instance.add(vid_name)
        # initionalize at the first time
        if not vid_name in segment.keys():
           segment[vid_name] = [[start_t, end_t, 1]]
        else:
           segment[vid_name].append([start_t, end_t, 1])

  # sort segments by start_time
  for vid in segment:
    segment[vid].sort(key=lambda x: x[0])
#source without label
  segment2={}
  for vid in segment:
    for k in range(len(segment[vid])):
      if k==0 and segment[vid][k][0]!=0.0:
        segment2[vid]=[[0.0,segment[vid][k][0],1]]
      if k!=0:
        if segment[vid][k][0]>segment[vid][k-1][1]:
          if not vid in segment2:
            segment2[vid]=[[segment[vid][k-1][1],segment[vid][k][0],1]]
          else:
            segment2[vid].append([segment[vid][k-1][1],segment[vid][k][0],1])
      if k==(len(segment[vid])-1):
        if not vid in segment2:
          segment2[vid]=[[segment[vid][k][1],segment[vid][k][1]+30,1]]
        else:
          segment2[vid].append([segment[vid][k][1],segment[vid][k][1]+30,1])

  if True:
      keys = list(segment2.keys())
      keys.sort()
      with open(os.path.join(split + '_segment_unlabel.txt'), 'w') as f:
        for k in keys:
          f.write("{}\n{}\n\n".format(k, segment2[k]))

  return segment2


if __name__ == '__main__':

    USE_FLIPPED = True
    print('Generate Source Training Segments')
    s_train_segment_unlabel = dataset_label(META_DIR + 'val', 'val')
    s_train_roidb_unlabel = generate_roidb('val', s_train_segment_unlabel)

    print (len(s_train_roidb_unlabel))
    print ("Save dictionary")
    pickle.dump(s_train_roidb_unlabel, open('train_data_25fps_flipped_s_u.pkl','wb'), pickle.HIGHEST_PROTOCOL)
