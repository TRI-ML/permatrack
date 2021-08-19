from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math

from ..video_dataset import VideoDataset

class PDTracking(VideoDataset):
  num_categories = 5
  dataset_folder = 'pd'
  default_resolution = [384, 960]
  class_name = ['Pedestrian', 'Car', 'Cyclist', 'Caravan/RV', 'Truck']
  # negative id is for "not as negative sample for abs(id)".
  # 0 for ignore losses for all categories in the bounding box region
  # ['Pedestrian', 'Car', 'Bicyclist', 'Bus', 'Caravan/RV',  'OtherMovable',
  # 'Motorcycle', 'Motorcyclist', 'OtherRider', 'Train', 'Truck', 'Dontcare']
  cat_ids = {1:1, 2:2, 3:3, 4:-9999, 5:4, 6:-2, 7:-9999, 8:-1, 9:-1, 10:-9999, 11:5}
  max_objs = 500
  def __init__(self, opt, split, rank=None):
    data_dir = os.path.join(opt.data_dir, self.dataset_folder)
    split_ = 'train' if opt.dataset_version != 'test' else 'test' #'test'
    img_dir = data_dir
    if split == 'train':
      ann_file_ = "train"
    else:
      ann_file_ = 'val'
    ann_path = os.path.join(
      data_dir, 'annotations', 'tracking_{}.json'.format(
        ann_file_))
    self.images = None
    super(PDTracking, self).__init__(opt, split, ann_path, img_dir)

    self.box_size_thresh = [300, 500, 300, 500, 500]

    if opt.only_ped:
      self.num_categories = 1
      self.class_name = ['person']
      self.cat_ids = {1:1, 2:-9999, 3:-1, 4:-9999, 5:-9999, 6:-9999, 7:-9999, 8:-1, 9:-1, 10:-9999, 11:-9999}
      self.box_size_thresh = [300]

    if opt.nu:
      self.num_categories = 8
      self.class_name = ['Car', 'Truck', 'Bus', 'Trailer', 'construction_vehicle', 'Pedestrian', 'Motorcycle', 'Bicycle']
      self.cat_ids =  {1:6, 2:1, 3:0, 4:3, 5:1, 6:-1, 7:-7, 8:0, 9:0, 10:-9999, 11:2, 12:5, 13:-8}
      self.box_size_thresh = [500, 500, 500, 500, 500, 300, 500, 500]

    self.alpha_in_degree = False
    self.depth_scale = 1
    self.dep_mask = 0
    self.dim_mask = 1
    self.rot_mask = 0
    self.amodel_offset_mask = 0
    self.ignore_amodal = True
    self.num_samples = len(self.images)
    self.exp_id = opt.exp_id
    if opt.const_v_over_occl:
      self.const_v_over_occl = True

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def save_results_ioueval(self, results, save_dir):
    formattted_results = []
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)

    for video in self.coco.dataset['videos']:
      video_id = video['id']
      images = self.video_to_images[video_id]
      
      for image_info in images:
        img_id = image_info['id']
        if not (img_id in results):
          continue
        frame_id = image_info['frame_id'] 
        for i in range(len(results[img_id])):
          item = results[img_id][i]
          if item['age'] != 1:
            continue
          if 'visibility' in item and not item['visibility']:
            continue
          category_id = item['class']
          track_id = item['tracking_id'] if 'tracking_id' in item else -1   
          bbox = [item['bbox'][0].item(), item['bbox'][1].item(), item['bbox'][2].item() - item['bbox'][0].item(), item['bbox'][3].item() - item['bbox'][1].item()]

          entry = {'video_id': video_id, 'image_id': img_id, 'category_id': category_id, 'track_id': track_id, 'bbox': bbox, 'score': item['score'].item()}
          formattted_results.append(entry)
    
    print(save_dir + '/iou_eval.json')
    json.dump(formattted_results, open(save_dir + '/iou_eval.json', 'w'))

  def run_eval(self, results, save_dir, write_to_file=False, dataset_version="val"):
    self.save_results_ioueval(results, save_dir)
    os.chdir("../tao")
    command = 'python scripts/evaluation/evaluate.py ' + \
              '../data/%s/annotations/tracking_%s_tao.json ' % (self.dataset_folder, dataset_version) + \
              '{}/iou_eval.json'.format(save_dir) + ' --config-updates CATEGORIES 1,2'

    if write_to_file:
      print("Writing to file")
      command += ' > ../exp/tracking/{}/eval_out.txt'.format(self.exp_id)
    os.system(command)

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))
