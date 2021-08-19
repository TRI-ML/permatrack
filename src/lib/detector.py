from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from progress.bar import Bar
import time
import torch
import math

from model.model import create_model, load_model
from model.decode import generic_decode
from model.utils import _nms
from model.utils import flip_tensor, flip_lr_off, flip_lr
from utils.utils import AverageMeter
from utils.image import get_affine_transform, affine_transform
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.post_process import generic_post_process
from utils.debugger import Debugger
from utils.tracker import Tracker
from dataset.dataset_factory import get_dataset
from trainer import ModleWithLoss, get_losses


class Detector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(
      opt.arch, opt.heads, opt.head_conv, opt=opt)
    self.model = load_model(self.model, opt.load_model, opt)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.opt = opt
    self.trained_dataset = get_dataset(opt.dataset)
    self.mean = np.array(
      self.trained_dataset.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(
      self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
    self.pause = not opt.no_pause
    self.rest_focal_length = self.trained_dataset.rest_focal_length \
      if self.opt.test_focal_length < 0 else self.opt.test_focal_length
    self.flip_idx = self.trained_dataset.flip_idx
    self.cnt = 0
    self.video_id = 0
    self.pre_images = None
    self.pre_hms = None
    self.h = None
    self.stream_test = opt.stream_test
    self.pre_image_ori = None
    self.tracker = Tracker(opt)
    self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)

    if self.opt.test_with_loss:
      self.loss_stats, self.loss = get_losses(opt)
      self.model_with_loss = ModleWithLoss(self.model, self.loss)
      self.model_with_loss.eval()
      self.avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}


  def run(self, image_or_path_or_tensor, meta={}):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
    self.debugger.clear()
    start_time = time.time()

    # read image
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      if 'image_path' in image_or_path_or_tensor.keys():
        image = cv2.imread(image_or_path_or_tensor['image_path'][0][0])
      else:
        image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []

    # for multi-scale testing
    for scale in self.opt.test_scales:
      scale_start_time = time.time()
      if not pre_processed:
        # not prefetch testing or demo
        images, meta = self.pre_process(image, scale, meta)
      else:
        # prefetch testing
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        if 'pre_dets' in pre_processed_images['meta']:
          meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
        if 'cur_dets' in pre_processed_images['meta']:
          meta['cur_dets'] = pre_processed_images['meta']['cur_dets']
      
      images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)

      # initializing tracker
      pre_inds = None

      if self.opt.tracking:
        # initialize the first frame
        if self.pre_images is None:
          print('Initialize tracking!')
          # Initialize previous image the same as the first frame
          self.pre_images = [images] * max(1, (self.opt.input_len - 1))
          self.tracker.init_track(
            meta['pre_dets'] if 'pre_dets' in meta else [])
          if self.opt.debug == 4:
            self.debugger.start_video(self.opt.debug_dir, self.video_id, image.shape[1], image.shape[0])
          self.video_id += 1
        pre_hm = None
        if self.opt.pre_hm:
          # render input heatmap from tracker status
          # pre_inds is not used in the current version.
          # We used pre_inds for learning an offset from previous image to
          # the current image.
          pre_hm, pre_inds = self._get_additional_inputs(
            self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
          if self.pre_hms is None:
            self.pre_hms = [pre_hm] * max(1, (self.opt.input_len - 1))
          else:
            self.pre_hms.append(pre_hm)
      
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      # run the network
      # output: the output feature maps, only used for visualizing
      # dets: output tensors after extracting peaks
      current_result = self.process(
        images, self.pre_images, self.pre_hms, pre_inds, return_time=True, original_batch=pre_processed_images)
      output = current_result['output']
      dets = current_result['dets']
      forward_time = current_result['forward_time']
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      # convert the cropped and 4x downsampled output coordinate system
      # back to the input image coordinate system
      result = self.post_process(dets, meta, scale)
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(result)

      if self.opt.debug >= 2:
        self.debug(
          self.debugger, images, result, output, scale, 
          pre_images=self.pre_images[-1] if not self.opt.no_pre_img else None, 
          pre_hms=pre_hm)

    # merge multi-scale testing results
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    
    if self.opt.tracking:
      # public detection mode in MOT challenge
      public_det = meta['cur_dets'] if self.opt.public_det else None
      # add tracking id to results
      results = self.tracker.step(results, public_det)
      self.pre_images.append(images)

    tracking_time = time.time()
    track_time += tracking_time - end_time
    tot_time += tracking_time - start_time

    if self.opt.debug >= 1:
      # image = cv2.resize(image, (images.size()[3], images.size()[2]))
      pred_hm = self.debugger.gen_colormap(_nms(output['hm'][0]).detach().cpu().numpy())
      pre_hm = None
      if self.opt.pre_hm:
        pre_hm = self.debugger.gen_colormap(self.pre_hms[-1].squeeze().unsqueeze(0).detach().cpu().numpy())
      self.show_results(self.debugger, image, results, pred_hm, pre_hm)
    self.cnt += 1

    show_results_time = time.time()
    display_time += show_results_time - end_time
    # return results and run time
    ret = {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time, 'track': track_time,
            'display': display_time}
    if self.opt.test_with_loss:
      ret['loss'] = current_result['loss']
      ret['loss_stats'] = current_result['loss_stats']
    if self.opt.save_video:
      try:
        # return debug image for saving video
        ret.update({'generic': self.debugger.imgs['generic']})
      except:
        pass
    return ret

  def close_video(self):
    self.debugger.stop_video()

  def _transform_scale(self, image, scale=1):
    '''
      Prepare input image in different testing modes.
        Currently support: fix short size/ center crop to a fixed size/ 
        keep original resolution but pad to a multiplication of 32
    '''
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_short > 0:
      if height < width:
        inp_height = self.opt.fix_short
        inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
      else:
        inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
        inp_width = self.opt.fix_short
      c = np.array([width / 2, height / 2], dtype=np.float32)
      s = np.array([width, height], dtype=np.float32)
    elif self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
      # s = np.array([inp_width, inp_height], dtype=np.float32)
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, c, s, inp_width, inp_height, height, width


  def pre_process(self, image, scale, input_meta={}):
    '''
    Crop, resize, and normalize image. Gather meta data for post processing 
      and tracking.
    '''

    resized_image, c, s, inp_width, inp_height, height, width = \
      self._transform_scale(image)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    out_height =  inp_height // self.opt.down_ratio
    out_width =  inp_width // self.opt.down_ratio
    trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
             if 'calib' in input_meta else \
             self._get_default_calib(width, height)}
    meta.update({'c': c, 's': s, 'height': height, 'width': width,
            'out_height': out_height, 'out_width': out_width,
            'inp_height': inp_height, 'inp_width': inp_width,
            'trans_input': trans_input, 'trans_output': trans_output})
    if 'pre_dets' in input_meta:
      meta['pre_dets'] = input_meta['pre_dets']
    if 'cur_dets' in input_meta:
      meta['cur_dets'] = input_meta['cur_dets']

    return images, meta


  def _trans_bbox(self, bbox, trans, width, height):
    '''
    Transform bounding boxes according to image crop.
    '''
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox


  def _get_additional_inputs(self, dets, meta, with_hm=True):
    '''
    Render input heatmap from previous trackings.
    '''
    trans_input, trans_output = meta['trans_input'], meta['trans_output']
    inp_width, inp_height = meta['inp_width'], meta['inp_height']
    out_width, out_height = meta['out_width'], meta['out_height']
    input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

    output_inds = []

    for det in dets:
      if det['score'] < self.opt.pre_thresh:
        continue
      if 'visibility' in det and det['visibility']  >= self.opt.visibility_thresh_eval:
        continue
      bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
      bbox_out = self._trans_bbox(
        det['bbox'], trans_output, out_width, out_height)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        # if with_hm and det['age'] <= 1:
        #   draw_umich_gaussian(input_hm[0], ct_int, radius)
        draw_umich_gaussian(input_hm[0], ct_int, radius)
        ct_out = np.array(
          [(bbox_out[0] + bbox_out[2]) / 2, 
           (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
        output_inds.append(ct_out[1] * out_width + ct_out[0])
    if with_hm:
      input_hm = input_hm[np.newaxis]
      if self.opt.flip_test:
        input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
      input_hm = torch.from_numpy(input_hm).to(self.opt.device)
    output_inds = np.array(output_inds, np.int64).reshape(1, -1)
    output_inds = torch.from_numpy(output_inds).to(self.opt.device)
    return input_hm, output_inds


  def _get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = output['hm'].sigmoid_()
    if 'visibility' in output:
      output['visibility'] = output['visibility'].sigmoid_()
    if 'hm_hp' in output:
      output['hm_hp'] = output['hm_hp'].sigmoid_()
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      output['dep'] *= self.opt.depth_scale
    return output


  def _flip_output(self, output):
    average_flips = ['hm', 'wh', 'dep', 'dim', 'visibility']
    neg_average_flips = ['amodel_offset']
    single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
      'hp_offset', 'rot', 'tracking', 'pre_hm']
    for head in output:
      if head in average_flips:
        output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
      if head in neg_average_flips:
        flipped_tensor = flip_tensor(output[head][1:2])
        flipped_tensor[:, 0::2] *= -1
        output[head] = (output[head][0:1] + flipped_tensor) / 2
      if head in single_flips:
        output[head] = output[head][0:1]
      if head == 'hps':
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
      if head == 'hm_hp':
        output['hm_hp'] = (output['hm_hp'][0:1] + \
          flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2

    return output


  def process(self, images, pre_images=None, pre_hms=None,
    pre_inds=None, return_time=False, original_batch=None):
    len_contex = max(1, self.opt.input_len-1)
    with torch.no_grad():
      torch.cuda.synchronize()
      if pre_images is not None and self.opt.is_recurrent:
          context = pre_images[-len_contex:]
          context_hms = [None] * len(context)
          if self.opt.pre_hm:
            context_hms = pre_hms[-len_contex:]
          batch_list = []
          if not self.stream_test:
            for i in range(len(context)):
              batch_list.append({'image': context[i], 'pre_hm': context_hms[i]})
          batch_list.append({'image': images, 'pre_hm': context_hms[-1]})

      elif not self.opt.is_recurrent:
        pre_images = pre_images[-1]

      if self.opt.test_with_loss:
        # if test with loss, add the annotation to the last dictionary in batch_list
        assert original_batch is not None
        for key in original_batch.keys():
          if key not in ['image','pre_hm', 'images', 'meta', 'is_first_frame', 'video_id', 'image_path']:
            batch_list[-1][key] = original_batch[key][0]
            if key != 'meta' and key!= 'gt_det':
              batch_list[-1][key] = batch_list[-1][key].to(self.opt.device)

      if self.stream_test:
        if self.opt.test_with_loss:
          output, loss, loss_stats, _, self.h = self.model_with_loss(batch_list, batch_size=1, stream=True, pre_gru_state=self.h, eval_mode=True)
        else:
          output, self.h = self.model.step(batch_list, self.h)
      else:
        if self.opt.test_with_loss:
          output, loss, loss_stats, pre_hms, _, _ = self.model_with_loss(batch_list, batch_size=1, eval_mode=True)
        else:
          output, _, _ = self.model(batch_list, pre_images, pre_hms)

      output = output[-1]
      output = self._sigmoid_output(output)
      output.update({'pre_inds': pre_inds})
      if self.opt.flip_test:
        output = self._flip_output(output)
      torch.cuda.synchronize()
      forward_time = time.time()
      
      dets = generic_decode(output, K=self.opt.K, opt=self.opt)

      torch.cuda.synchronize()
      for k in dets:
        dets[k] = dets[k].detach().cpu().numpy()
    return_dict = {
    'output': output,
    'dets': dets
      }
    if return_time:
      return_dict.update({
        'forward_time': forward_time
        })
    if self.opt.test_with_loss:
      return_dict.update({
        'loss': loss,
        'loss_stats': loss_stats
        })
    return return_dict

  def post_process(self, dets, meta, scale=1):
    dets = generic_post_process(
      self.opt, dets, [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'], self.opt.num_classes,
      [meta['calib']], meta['height'], meta['width'])
    self.this_calib = meta['calib']
    
    if scale != 1:
      for i in range(len(dets[0])):
        for k in ['bbox', 'hps']:
          if k in dets[0][i]:
            dets[0][i][k] = (np.array(
              dets[0][i][k], np.float32) / scale).tolist()
    return dets[0]

  def merge_outputs(self, detections):
    assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
    results = []
    for i in range(len(detections[0])):
      if detections[0][i]['score'] > self.opt.out_thresh:
        results.append(detections[0][i])
    return results

  def debug(self, debugger, images, dets, output, scale=1, 
    pre_images=None, pre_hms=None):
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if 'hm_hp' in output:
      pred = debugger.gen_colormap_hp(
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')

    if pre_images is not None:
      pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
      pre_img = np.clip(((
        pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
      debugger.add_img(pre_img, 'pre_img')
      if pre_hms is not None:
        pre_hm = debugger.gen_colormap(
          pre_hms[0].detach().cpu().numpy())
        debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')


  def show_results(self, debugger, image, results, hm, pre_hm=None):
    debugger.add_img(image, img_id='generic')
    if self.opt.tracking:
      debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image, 
        img_id='previous')
      self.pre_image_ori = image

    if pre_hm is not None:
      debugger.add_blend_img(image, pre_hm, 'generic')

    if self.opt.vis_hms:
      debugger.add_blend_img(image, hm, 'pred_hm')
    
    for j in range(len(results)):
      if results[j]['score'] > self.opt.vis_thresh:
        item = results[j]
        if ('bbox' in item):
          sc = item['score'] if self.opt.demo == '' or \
            not ('tracking_id' in item) else item['tracking_id']
          sc = item['tracking_id'] if self.opt.show_track_color else sc
          c = None
          # if 'visibility' in item and not item['visibility']:
          #   c = (211,211,211)
          
          debugger.add_coco_bbox(
            item['bbox'], item['class'] - 1, sc, img_id='generic', c=c)

        if 'tracking' in item:
          debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')
        
        tracking_id = item['tracking_id'] if 'tracking_id' in item else -1
        if 'tracking_id' in item and self.opt.demo == '' and \
          not self.opt.show_track_color:
          debugger.add_tracking_id(
            item['ct'], item['tracking_id'], img_id='generic')

        if (item['class'] in [1, 2]) and 'hps' in item:
          debugger.add_coco_hp(item['hps'], tracking_id=tracking_id,
            img_id='generic')

    if len(results) > 0 and \
      'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
      debugger.add_3d_detection(
        image if not self.opt.qualitative else cv2.resize(
          debugger.imgs['pred_hm'], (image.shape[1], image.shape[0])), 
        False, results, self.this_calib,
        vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
      debugger.add_bird_view(
        results, vis_thresh=self.opt.vis_thresh,
        img_id='bird_pred', cnt=self.cnt)
      if self.opt.show_track_color and self.opt.debug == 4:
        del debugger.imgs['generic'], debugger.imgs['bird_pred']
    if 'ddd_pred' in debugger.imgs:
      debugger.imgs['generic'] = debugger.imgs['ddd_pred']
    if self.opt.debug == 4:
      # debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))
      if self.opt.vis_hms:
        debugger.add_to_video(vis_type='pred_hm')
      else:
        debugger.add_to_video()
    else:
      debugger.show_all_imgs(pause=self.pause)
  

  def reset_tracking(self):
    self.tracker.reset()
    self.pre_images = None
    self.pre_hms = None
    self.pre_image_ori = None
    self.h = None
