from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from model.decode import generic_decode

from utils.image import gaussian_radius, draw_umich_gaussian

from model.ConvGRU import ConvGRU

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class ParallelModel(nn.Module):
  def __init__(self, conv, head1, head2):
    super(ParallelModel, self).__init__()

    self.conv = conv
    self.head1 = head1
    self.head2 = head2

  def forward(self, x):
    hidden = F.relu(self.conv(x))

    return self.head1(hidden), self.head2(hidden)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.opt = opt

        if opt.is_recurrent:
          in_channel = last_channel
          if opt.pre_hm:
              in_channel += 1
          self.conv_gru = ConvGRU(in_channel, last_channel, (opt.gru_filter_size, opt.gru_filter_size), opt.num_gru_layers, batch_first=True, nl=opt.nl)

        self.heads = heads
        for head in self.heads:
          classes = self.heads[head]
          head_conv = head_convs[head]
          if len(head_conv) > 0:
            out = nn.Conv2d(head_conv[-1], classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(last_channel, head_conv[0],
                            kernel_size=head_kernel, 
                            padding=head_kernel // 2, bias=True)
            convs = [conv]
            for k in range(1, len(head_conv)):
              convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                            kernel_size=1, bias=True))
            if len(convs) == 1:
              fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
            elif len(convs) == 2:
              fc = nn.Sequential(
                convs[0], nn.ReLU(inplace=True), 
                convs[1], nn.ReLU(inplace=True), out)
            elif len(convs) == 3:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), out)
            elif len(convs) == 4:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), 
                  convs[3], nn.ReLU(inplace=True), out)
            if 'hm' in head or 'visibility' in head:
              if not isinstance(fc, ParallelModel):
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fc.head1.bias.data.fill_(opt.prior_bias)
                fc.head2.bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)
          else:
            fc = nn.Conv2d(last_channel, classes, 
                kernel_size=1, stride=1, padding=0, bias=True)
            if 'hm' in head:
              fc.bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)
          self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError

    def freeze_backbone(self):
      raise NotImplementedError

    def freeze_gru(self):
      for parameter in self.conv_gru.parameters():
          parameter.requires_grad = False    
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def step(self, x, h):
      feats = self.imgpre2feats(x, None, torch.zeros(1))

      batch_size = int(len(feats[0]))
      inp = feats[0].view(batch_size, 1, feats[0].size(1), feats[0].size(2), feats[0].size(3))

      curr_step = inp[:, 0, :, :, :]
      if self.opt.pre_hm:
        hm = nn.functional.interpolate(x[0]['pre_hm'], size=(inp.size(3), inp.size(4)), mode="bilinear")
        curr_step = torch.cat((curr_step, hm), 1).unsqueeze(1)
      else:
        curr_step = curr_step.unsqueeze(1)
      intermediate_outputs, layer_reset_list, layer_update_list, last_output = self.conv_gru(curr_step, h)
      h = last_output
      feats = last_output[-1:][0]

      out = []
      z = self.apply_heads(feats, {})
      out.append(z)

      return out, h

    def apply_heads(self, feature, z):
      for head in self.heads:       
        z[head] = self.__getattr__(head)(feature)

      return z

    def forward(self, x, pre_img=None, pre_hm=None, batch_size=1):
      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        feats = self.img2feats(x)
      
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
            z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks): # zero for GRU model
          if self.opt.is_recurrent:

            input_len = int(len(feats[s]) / batch_size)
            inp = feats[s].view(batch_size, input_len, feats[s].size(1), feats[s].size(2), feats[s].size(3))
            hm = torch.zeros(batch_size, 1, inp.size(3), inp.size(4)).cuda()
            h = None
            pre_hms = []
            # process a batch of frames one by one
            for i in range(inp.size(1)):
              curr_step = inp[:, i, :, :, :]
              if self.opt.pre_hm:
                curr_step = torch.cat((curr_step, hm), 1)
              intermediate_outputs, layer_reset_list, layer_update_list, last_output = self.conv_gru(curr_step.unsqueeze(1), h)
              h = last_output
              feats[s] = last_output[-1:][0]

              z = {}
              z = self.apply_heads(feats[s], z)
              out.append(z)

              pre_hms.append(hm)
              if self.opt.pre_hm and i < (inp.size(1) - 1):
                if type(x) == list:
                  hm = nn.functional.interpolate(x[i + 1]['pre_hm'], size=(inp.size(3), inp.size(4)), mode="bilinear")
                else:
                  hm = nn.functional.interpolate(pre_hm, size=(inp.size(3), inp.size(4)), mode="bilinear")
          else:
            z = self.apply_heads(feats[s], {})
          
            out.append(z)
            pre_hms = [pre_hm]

      return out, pre_hms, x
