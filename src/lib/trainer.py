from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
import torch
import numpy as np
from progress.bar import Bar

from model.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss, WeightedBCELossNoLogits
from model.matcher import HungarianMatcher
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import generic_post_process
from model.utils import _tranpose_and_gather_feat

class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss(opt=opt)
    self.crit_reg = RegWeightedL1Loss()
    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()
    if 'nuscenes_att' in opt.heads:
      self.crit_nuscenes_att = WeightedBCELoss()
    self.opt = opt

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'visibility' in output:
      output['visibility'] = _sigmoid(output['visibility'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output

  def forward(self, outputs, batch, input_len=None, frame_ind=1):
    opt = self.opt
    losses = {head: 0 for head in opt.heads}

    for s in range(opt.num_stacks):
      output = outputs[s]
      output = self._sigmoid_output(output)

      if 'hm' in output:
        tot_hm_loss = self.crit(
          output['hm'], batch['hm'], batch['ind'], 
          batch['mask'], batch['cat'])
        losses['hm'] += tot_hm_loss / opt.num_stacks

      if 'visibility' in output:
        tot_vis_loss = self.crit(
          output['visibility'], batch['visibility'], batch['visibility_ind'], 
          batch['visibility_mask'], batch['visibility_cat'])
        losses['visibility'] += tot_vis_loss / opt.num_stacks
      
      regression_heads = [
        'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps', 
        'dep', 'dim', 'amodel_offset', 'velocity']

      for head in regression_heads:
        if head in output:
          losses[head] += self.crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks
      
      if 'hm_hp' in output:
        losses['hm_hp'] += self.crit(
          output['hm_hp'], batch['hm_hp'], batch['hp_ind'], 
          batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
        if 'hp_offset' in output:
          losses['hp_offset'] += self.crit_reg(
            output['hp_offset'], batch['hp_offset_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
        
      if 'rot' in output:
        losses['rot'] += self.crit_rot(
          output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres']) / opt.num_stacks

      if 'nuscenes_att' in output:
        losses['nuscenes_att'] += self.crit_nuscenes_att(
          output['nuscenes_att'], batch['nuscenes_att_mask'],
          batch['ind'], batch['nuscenes_att']) / opt.num_stacks

    losses['tot'] = 0
    for head in opt.heads:
      losses['tot'] += opt.weights[head] * losses[head]
    
    return losses['tot'], losses

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, batch, batch_size=1, stream=False, pre_gru_state=None, eval_mode=False):
    """ Forward function

    Parameters
    ----------
    batch: list or dict
      Dict: legacy input format from CenterNet. To be deprecated.
      List: A list of dictionary with input and annotations.

    batch_size: int
      Batch size

    stream: bool
      Whether the model is evaluated in steam mode.

    pre_gru_state: torch.Tensor
      Previous ConvGRU state vector.

    eval_mode: bool
      whether it is used for evaluation.
    """

    if type(batch) != list:
      pre_img = batch['pre_img'] if 'pre_img' in batch else None
      pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
      outputs, pre_hm, _ = self.model(batch['image'], pre_img, pre_hm, batch_size)
      loss, loss_stats = self.loss([outputs[-1]], batch)
    else:
      if eval_mode:
        pre_img = []
        pre_hm = []
        for i in range(len(batch)-1):
          pre_img.append(batch[i]['image'])
          pre_hm.append(batch[i]['pre_hm'])
      else:
        pre_img = torch.zeros(5, 5)
        pre_hm = None

      if stream and eval_mode:
        outputs, output_gru_state = self.model.step(batch, pre_gru_state)
      else:
        outputs, pre_hm, batch = self.model(batch, pre_img, pre_hm, batch_size)
      loss = None
      stats = []

      if not eval_mode:
        for i in range(len(batch)):
          loss_step, loss_stats = self.loss([outputs[i]], batch[i], len(batch), i)
          stats.append(loss_stats)
          if loss is None:
            loss = loss_step
          else:
            loss += loss_step
        loss /= len(batch)
      else:
          loss_step, loss_stats = self.loss([copy.deepcopy(outputs[-1])], batch[-1])
          stats.append(loss_stats)
          loss = loss_step

    if stream:
      return outputs, loss, loss_stats, batch, output_gru_state
    return outputs, loss, loss_stats, pre_hm, batch

def get_losses(opt):
  loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
      'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity', 'visibility']
  loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
  loss = GenericLoss(opt)
  return loss_states, loss

class Trainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    self.rank = device

    self.model_with_loss = DDP(self.model_with_loss, device_ids=[device], find_unused_parameters=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    len_data = len(data_loader)
    num_iters = len_data
    if phase == 'train':
      num_iters = len_data if opt.num_iters < 0 else opt.num_iters
    else:
      num_iters = len_data if opt.num_val_iters < 0 else opt.num_val_iters

    if self.rank == 0:
      bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    real_iter = iter(data_loader)
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break

      data_time.update(time.time() - end)

      if type(batch) == list:
        batch_size = batch[0]['image'].size(0)
        for i in range(len(batch)):
          for k in batch[i]:
            if k != 'meta' and k!= 'gt_det' and k!= 'image_path':
              batch[i][k] = batch[i][k].to(device=opt.device, non_blocking=True)     
      else:
        for k in batch:
          if k != 'meta' and k!= 'gt_det' and k!= 'image_path':
            batch[k] = batch[k].to(device=opt.device, non_blocking=True) 

      outputs, loss, loss_stats, pre_hms, batch = model_with_loss(batch, batch_size)  
      output = outputs[-1]
      prev_output = outputs[0]
        
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      batch_time.update(time.time() - end)
      end = time.time()

      if self.rank == 0:
        Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
          epoch, iter_id, num_iters, phase=phase,
          total=bar.elapsed_td, eta=bar.eta_td)
        for l in avg_loss_stats:
          avg_loss_stats[l].update(
            loss_stats[l].mean().item(), batch_size)
          Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        if opt.print_iter > 0: # If not using progress bar
          if iter_id % opt.print_iter == 0:
            print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
        else:
          bar.next()
      
      del output, loss, loss_stats
    
    if self.rank == 0:
      bar.finish()
    
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    if self.rank == 0:
      ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
