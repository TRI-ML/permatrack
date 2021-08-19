from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import copy

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from dataset.dataset_factory import dataset_factory
from detector import Detector
import math


class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.pre_process_func = pre_process_func
    self.get_default_calib = dataset.get_default_calib
    self.opt = opt
  
  def __getitem__(self, index):

    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    image = cv2.imread(img_path)

    images, meta = {}, {}
    for scale in opt.test_scales:
      input_meta = {}
      calib = img_info['calib'] if 'calib' in img_info \
        else self.get_default_calib(image.shape[1], image.shape[0])
      input_meta['calib'] = calib
      images[scale], meta[scale] = self.pre_process_func(
        image, scale, input_meta)
    ret = {'images': images, 'image': image, 'meta': meta}
    if 'frame_id' in img_info and img_info['frame_id'] == 1:
      ret['is_first_frame'] = 1
      ret['video_id'] = img_info['video_id']
    return img_id, ret

  def __len__(self):
    return len(self.images)

def prefetch_test(opt):
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)
  
  if opt.load_results != '':
    load_results = json.load(open(opt.load_results, 'r'))
    for img_id in load_results:
      for k in range(len(load_results[img_id])):
        if load_results[img_id][k]['class'] - 1 in opt.ignore_loaded_cats:
          load_results[img_id][k]['score'] = -1
  else:
    load_results = {}

  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

  results = {}
  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  if opt.use_loaded_results:
    for img_id in data_loader.dataset.images:
      results[img_id] = load_results['{}'.format(img_id)]
    num_iters = 0
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):

    if ind >= num_iters:
      break
    if opt.tracking and ('is_first_frame' in pre_processed_images):
      if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
        pre_processed_images['meta']['pre_dets'] = \
          load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
      else:
        print('No pre_dets for', int(img_id.numpy().astype(np.int32)[0]), 
          '. Use empty initialization.')
        pre_processed_images['meta']['pre_dets'] = []
      detector.reset_tracking()
      print('Start tracking video', int(pre_processed_images['video_id']))
    if opt.public_det:
      if '{}'.format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
        pre_processed_images['meta']['cur_dets'] = \
          load_results['{}'.format(int(img_id.numpy().astype(np.int32)[0]))]
      else:
        print('No cur_dets for', int(img_id.numpy().astype(np.int32)[0]))
        pre_processed_images['meta']['cur_dets'] = []

    ret = detector.run(pre_processed_images)
    results[int(img_id.numpy().astype(np.int32)[0])] = ret['results']
    
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    if opt.print_iter > 0:
      if ind % opt.print_iter == 0:
        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
    else:
      bar.next()
  bar.finish()
  if opt.save_results:
    print('saving results to', opt.save_dir + '/save_results_{}{}.json'.format(
      opt.test_dataset, opt.dataset_version))
    json.dump(_to_list(copy.deepcopy(results)), 
              open(opt.save_dir + '/save_results_{}{}.json'.format(
                opt.test_dataset, opt.dataset_version), 'w'))
  dataset.run_eval(results, opt.save_dir, opt.write_to_file, opt.dataset_version)

def test_with_loss(opt):
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)

  split = 'val' if not opt.trainval else 'test'
  dataset_opt = copy.deepcopy(opt)
  # in val/test mode, data will come in sequence.
  # There is no need to pad the data in dataset level.
  dataset_opt.input_len = 1
  dataset = Dataset(dataset_opt, split)
  detector = Detector(opt)

  load_results = {}

  data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=1, shuffle=False, num_workers=4,
      pin_memory=True)

  results = {}
  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'track']
  avg_time_stats = {t: AverageMeter() for t in time_stats}

  avg_loss_stats = {l: AverageMeter() for l in detector.loss_stats}
  for ind, batch in enumerate(data_loader):
    img_id = dataset.images[ind]

    # convert batch into key first structure
    adapted_batch = {}
    for key in batch[0].keys():
        adapted_batch[key] = [x[key] for x in batch if key in x]
    image_list = adapted_batch['image']
    adapted_batch['images'] = {opt.test_scales[0]:image_list}
    meta_dict = adapted_batch['meta'][0]
    adapted_batch['meta']={opt.test_scales[0]:meta_dict}
    adapted_batch['image'] = image_list[0].permute([0,2,3,1])


    if ind >= num_iters:
      break
    if opt.tracking and (1 in adapted_batch['frame_id']):
      if img_id in load_results:
        adapted_batch['meta']['pre_dets'] = \
          load_results[img_id]
      else:
        print('No pre_dets for', img_id,
          '. Use empty initialization.')
        adapted_batch['meta']['pre_dets'] = []
      detector.reset_tracking()
      print('Start tracking video', int(adapted_batch['video_id'][0]))
    if opt.public_det:
      if img_id in load_results:
        adapted_batch['meta']['cur_dets'] = \
          load_results[img_id]
      else:
        print('No cur_dets for', img_id)
        adapted_batch['meta']['cur_dets'] = []

    ret = detector.run(adapted_batch)
    results[img_id] = ret['results']

    # record loss results
    for l in avg_loss_stats:
      avg_loss_stats[l].update(
          ret['loss_stats'][l].mean().item(), 1)

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    if opt.print_iter > 0:
      if ind % opt.print_iter == 0:
        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
    else:
      bar.next()
  bar.finish()

  # write loss result to json
  loss_ret = {k: v.avg for k, v in avg_loss_stats.items()}
  json.dump(loss_ret, open("../exp/tracking/{}/eval_loss_out.json".format(opt.exp_id), "w"))
  if opt.save_results:
    print('saving results to', opt.save_dir + '/save_results_{}{}.json'.format(
      opt.test_dataset, opt.dataset_version))
    json.dump(_to_list(copy.deepcopy(results)),
              open(opt.save_dir + '/save_results_{}{}.json'.format(
                opt.test_dataset, opt.dataset_version), 'w'))

  dataset.run_eval(results, opt.save_dir, opt.write_to_file, opt.dataset_version)

def test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = dataset_factory[opt.test_dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  
  split = 'val' if not opt.trainval else 'test'
  dataset = Dataset(opt, split)
  detector = Detector(opt)

  if opt.load_results != '': # load results in json
    load_results = json.load(open(opt.load_results, 'r'))

  results = {}
  num_iters = len(dataset) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind in range(num_iters):
    img_id = dataset.images[ind]
    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    input_meta = {}
    if 'calib' in img_info:
      input_meta['calib'] = img_info['calib']
    if (opt.tracking and ('frame_id' in img_info) and img_info['frame_id'] == 1):
      detector.reset_tracking()
      input_meta['pre_dets'] = load_results[img_id]

    ret = detector.run(img_path, input_meta)    
    results[img_id] = ret['results']

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()
  bar.finish()
  detector.close_video()
  if opt.save_results:
    print('saving results to', opt.save_dir + '/save_results_{}{}.json'.format(
      opt.test_dataset, opt.dataset_version))
    json.dump(_to_list(copy.deepcopy(results)), 
              open(opt.save_dir + '/save_results_{}{}.json'.format(
                opt.test_dataset, opt.dataset_version), 'w'))
  dataset.run_eval(results, opt.save_dir, opt.write_to_file, opt.dataset_version)


def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().parse()

  if opt.test_with_loss:
    test_with_loss(opt)
  elif opt.not_prefetch_test:
    test(opt)
  else:
    prefetch_test(opt)
