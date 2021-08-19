from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .datasets.mot import MOT
from .datasets.crowdhuman import CrowdHuman
from .datasets.kitti_tracking import KITTITracking
from .datasets.pd_tracking import PDTracking
from .datasets.custom_dataset import CustomDataset
from .datasets.nuscenes_tracking import nuScenesTracking

dataset_factory = {
  'custom': CustomDataset,
  'mot': MOT,
  'crowdhuman': CrowdHuman,
  'kitti_tracking': KITTITracking,
  'pd_tracking': PDTracking,
  'nuscenes_tracking': nuScenesTracking
}


def get_dataset(dataset):
  return dataset_factory[dataset]
