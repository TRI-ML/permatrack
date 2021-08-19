from .generic_dataset import GenericDataset
import os
import numpy as np
import random
from utils.image import get_affine_transform, affine_transform
import copy
import torch.nn.functional as F
import torch
from torchvision.ops.boxes import box_iou
from utils.pose import Pose

class VideoDataset(GenericDataset):
    
    def __init__(self, opt=None, split=None, ann_path=None, img_dir=None):
        self.occlusion_thresh = opt.occlusion_thresh
        self.visibility_thresh = opt.visibility_thresh
        self.input_len = None
        self.min_frame_dist = None
        self.max_frame_dist = None
        self.box_size_thresh = None
        self.min_frame_dist = None
        self.max_frame_dist = None
        self.const_v_over_occl = False
        super(VideoDataset, self).__init__(opt, split, ann_path, img_dir)
        if split != 'train':
            self.input_len = 1

    def hide_occlusions(self, anns):
        for i, frame_anns in enumerate(anns):
            for ann in frame_anns:
                if (ann['occlusion'] < self.occlusion_thresh) and (ann['occlusion'] > self.visibility_thresh):
                    ann['iscrowd'] = 1
                elif ann['occlusion'] <= self.visibility_thresh:
                    ann['iscrowd'] = 2

        return anns

    def project_3dbbox(self, center, intrinsics):
        bbox_proj = np.dot(np.array(intrinsics), np.array(center))
        x = bbox_proj[0] / bbox_proj[2]
        y = bbox_proj[1] / bbox_proj[2]
        return 0.5 * x, np.max(0.5 * y - 158, 0)

    def process_occlusions(self, anns, img_infos, trans_out, h, w, flipped=False):
        # visibility status of every track
        track_vis = {}
        # annotations of every track
        tracks = {}
        # filtered list of visible annotations
        vis_anns = []
        # filtred list of invsible annotations
        invis_anns = []
        # minimal number of consequitive visible frames after which to start supervising behind occlusions
        visibility_range = 2
        # aggregate tracks info for a clip
        for i, frame_anns in enumerate(anns):
            filtered_frame_anns = []
            img_info = img_infos[i]
            if 'pose_quat' in img_info:
                scene_pose = Pose(np.array(img_info['pose_quat']), np.array(img_info['pose_tvec']))

            for ann in frame_anns:
                if ann['track_id'] not in tracks:
                    tracks[ann['track_id']] = [None] * len(anns)
                 
                if 'pose_quat' in ann:
                    ann['pose'] = Pose(np.array(ann['pose_quat']), np.array(ann['pose_tvec']))
                    ann['pose'] = scene_pose * ann['pose']

                tracks[ann['track_id']][i] = ann

                # calculate 2d and 3d velocity for the current frame
                if i > 0 and 'pose' in ann:
                    ann = self.assign_speed(ann, tracks[ann['track_id']][i - 1])
                    ann = self.assign_3d_speed(ann, tracks[ann['track_id']][i - 1])
                else:
                    ann['v'] = np.zeros(2, np.float32)
                    ann['3dv'] = np.zeros(3, np.float32)

                occlusion_thresh = self.occlusion_thresh
                track_id = ann['track_id']
                # object is visible unless it is occluded
                visible = True
                if track_id not in track_vis:
                    track_vis[track_id] = [None] * len(anns)
                if ann['occlusion'] < occlusion_thresh:
                    visible = False

                # tiny boxes are treated as invisible
                if 'modal_bbox' in ann:
                    modal_bbox = ann['modal_bbox']
                    if flipped:
                        modal_bbox = [w - modal_bbox[0] - 1 - modal_bbox[2], modal_bbox[1], modal_bbox[2], modal_bbox[3]]
                    bbox, _, truncated = self._get_bbox_output(modal_bbox, trans_out, h, w)
                    box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if box_h * box_w < 10:
                        visible = False

                track_vis[track_id][i] = visible

                # for the first "visibility_range" frames instanteneous visibility determines the visibility status of an object
                if (i < visibility_range) and visible:
                    filtered_frame_anns.append(ann)
                elif (i < visibility_range) and not visible and (ann['occlusion'] > self.visibility_thresh):
                    ann['iscrowd'] = 1
                    filtered_frame_anns.append(ann)
                elif (i < visibility_range) and (ann['occlusion'] <= self.visibility_thresh):
                    ann['iscrowd'] = 2
                
            if i < visibility_range:
                vis_anns.append(filtered_frame_anns)
                invis_anns.append([])

        # computing visiblity status for the objects in the remaining frames
        for i in range(visibility_range, len(anns)):
            frame_anns = anns[i]
            vis_frame_anns = []
            invis_frame_anns = []
            for ann in frame_anns:
                track_id = ann['track_id']

                # if an object is not occluded it is treated as visible
                if track_vis[track_id][i]:
                    vis_frame_anns.append(ann)
                    continue

                # occluded objects are supervised if they have been visible for at least "visibility_range" consequitive frames in the past
                previously_seen = track_vis[track_id][i - visibility_range]
                for j in range(visibility_range - 1, 0, -1):
                    previously_seen = previously_seen and track_vis[track_id][i - j]
                if previously_seen:
                    track_vis[track_id][i] = True
                    invis_frame_anns.append(ann)
                # partially occluded objects are ignored if the have not been seen in the past
                elif ann['occlusion'] > self.visibility_thresh:
                    ann['iscrowd'] = 1
                    vis_frame_anns.append(ann)

            vis_anns.append(vis_frame_anns)
            invis_anns.append(invis_frame_anns)
        
        return vis_anns, invis_anns

    def get_ann_by_id(self, anns, track_id):
        for ann in anns:
            if ann['track_id'] == track_id:
                return ann
                
        return None

    def assign_speed(self, ann, prev_ann):
        if prev_ann is None:
            v = np.zeros(2, np.float32)
        else:
            bbox = self._coco_box_to_bbox(ann['bbox']).copy()
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

            bbox = self._coco_box_to_bbox(prev_ann['bbox']).copy()
            prev_ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

            v = ct - prev_ct

        ann['v'] = v

        return ann

    def assign_3d_speed(self, ann, prev_ann):
        if prev_ann is None:
            v = np.zeros(3, np.float32)
        else:
            v = ann['pose'].tvec - prev_ann['pose'].tvec

        ann['3dv'] = v

        return ann

    def __getitem__(self, index):
        opt = self.opt
        if self.input_len == None:
            self.input_len = opt.input_len
        imgs, anns, img_infos, img_paths = self._load_data(index, self.input_len)

        height, width = imgs[0].shape[0], imgs[0].shape[1]
        c = np.array([imgs[0].shape[1] / 2., imgs[0].shape[0] / 2.], dtype=np.float32)
        s = max(imgs[0].shape[0], imgs[0].shape[1]) * 1.0 if not self.opt.not_max_crop \
        else np.array([imgs[0].shape[1], imgs[0].shape[0]], np.float32)
        aug_s, rot, flipped = 1, 0, 0
        if self.split == 'train':
            c, aug_s, rot = self._get_aug_param(c, s, width, height)
        s = s * aug_s
        if np.random.random() < opt.flip:
            flipped = 1

        trans_input = get_affine_transform(
        c, s, rot, [self.default_resolution[1], self.default_resolution[0]])
        trans_output = get_affine_transform(
        c, s, rot, [self.default_resolution[1] // self.opt.down_ratio, self.default_resolution[0] // self.opt.down_ratio])
 
        # if there are occlusion labels in the annotations, separate them into visible and invisible ones, or simply filter out invisible annotations
        invis_anns = [[]] * len(anns)
        if (len(anns[0]) > 0) and 'occlusion' in anns[0][0] and opt.sup_invis:
            anns, invis_anns = self.process_occlusions(copy.deepcopy(anns), img_infos, trans_output, height, width, flipped)
        elif (len(anns[0]) > 0) and 'occlusion' in anns[0][0] and not opt.sup_invis:
            anns = self.hide_occlusions(copy.deepcopy(anns))

        rets = []
        pre_anns = copy.deepcopy(anns[0])
        pre_invis_anns = copy.deepcopy(anns[0])
        apply_noise_to_centers = False
        skipped_objs = {}
        # invisible objects that have left the frame according to cosntant velocity assumption
        out_of_frame = set([])
        # iterate over frames
        for i in range(len(imgs)):
            # do not apply noise to previous centers in the first frame to avoid non-zero tracking targets
            if i > 0:
                apply_noise_to_centers = True
            img = imgs[i]
            anns_frame = copy.deepcopy(anns[i])
            anns_invis_frame = copy.deepcopy(invis_anns[i])
            img_info = img_infos[i]
            if flipped:
                img = img[:, ::-1, :]
                anns_frame = self._flip_anns(anns_frame, width)
                anns_invis_frame = self._flip_anns(anns_invis_frame, width)
                pre_anns = self._flip_anns(pre_anns, width)
                pre_invis_anns = self._flip_anns(pre_invis_anns, width)

            pre_invis_anns = self.update_pre(pre_invis_anns, skipped_objs)
            pre_anns.extend(pre_invis_anns)

            pre_hm, pre_cts, track_ids, occl_lengths, pre_ids = self._get_pre_dets(
                pre_anns, trans_input, trans_output, apply_noise_to_centers)

            if i > 0 and not self.same_aug_pre and self.split == 'train':
                c, aug_s, _ = self._get_aug_param(c, s, width, height, disturb=True)
                s = s * aug_s
                trans_input = get_affine_transform(
                  c, s, rot, [self.default_resolution[1], self.default_resolution[0]])
                trans_output = get_affine_transform(
                  c, s, rot, [self.default_resolution[1] // self.opt.down_ratio, self.default_resolution[0] // self.opt.down_ratio])

            inp = self._get_input(img, trans_input, self.mean, self.std)
            ret = {'image': inp}
            gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}
            
            ### init samples
            self._init_ret(ret, gt_det)
            calib = self._get_calib(img_info, width, height)
            # extract camera pose and intrinsics if needed
            if self.opt.const_v_over_occl and 'pose_quat' in img_info:
                scene_pose = Pose(np.array(img_info['pose_quat']), np.array(img_info['pose_tvec']))
                scene_intrinsics = img_info['intrinsics']
            
            num_objs = min(len(anns_frame), self.max_objs)
            counter = 0
            # process visible objects first
            for k in range(num_objs):
                ann = anns_frame[k]
                cls_id = int(self.cat_ids[ann['category_id']])
                if cls_id > self.opt.num_classes or cls_id <= -999:
                    continue
                bbox, bbox_amodal, truncated = self._get_bbox_output(
                    ann['bbox'], trans_output, height, width)
                orig_box = ann['bbox']
                box_size = orig_box[2] * orig_box[3]
                if ('iscrowd' in ann) and (ann['iscrowd'] == 2):
                    continue

                if ann['track_id'] in out_of_frame:
                    out_of_frame.remove(ann['track_id'])
                elif (cls_id <= 0) or (('iscrowd' in ann) and (ann['iscrowd'] == 1)) or (box_size < self.box_size_thresh[cls_id - 1]):
                    v = self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids)
                    continue

                is_added = self._add_instance(
                    ret, gt_det, counter, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
                    calib, pre_cts, track_ids, pre_ids, False, occl_lengths)
                if is_added:
                    counter += 1

            num_objs = min(len(anns_invis_frame), self.max_objs)
            # last seen annotations of objects that are currently occluded
            prev_skipped_objs = skipped_objs
            skipped_objs = {}
            # process invisible objects
            for k in range(num_objs):
                ann = anns_invis_frame[k]

                bbox, bbox_amodal, truncated = self._get_bbox_output(
                        ann['bbox'], trans_output, height, width)
                orig_box = ann['bbox']
                box_size = orig_box[2] * orig_box[3]

                cls_id = int(self.cat_ids[ann['category_id']])
                if cls_id > self.opt.num_classes or cls_id <= 0:
                    continue

                # propagate object box with constant velocity
                if self.const_v_over_occl:
                    if ann['track_id'] in out_of_frame:
                        continue

                    gt_box = ann['bbox']

                    if ann['track_id'] in prev_skipped_objs:
                        ann = prev_skipped_objs[ann['track_id']]
                    else:
                        # first frame of an occlusion, find annotation of that object in the previous frame (last seen)
                        ann = self.get_ann_by_id(pre_anns, ann['track_id'])
                        ann['last_seen_size'] = ann['bbox'][2] * ann['bbox'][2] 
                        if ann['track_id'] not in track_ids:
                            out_of_frame.add(ann['track_id'])
                            continue

                    if 'occl_length' not in ann:
                        ann['occl_length'] = 1

                    last_seen_size = ann['last_seen_size']

                    # propagate box center
                    if self.opt.const_v_2d:
                        curr_x = ann['bbox'][0] + ann['bbox'][2] / 2 
                        curr_y = ann['bbox'][1] + ann['bbox'][3] / 2 

                        curr_x += ann['v'][0]
                        curr_y += ann['v'][1]
                    else:
                        ann['pose'].tvec += ann['3dv']
                        frame_3dpose = scene_pose.inverse() * ann['pose'] 
                        curr_x, curr_y = self.project_3dbbox(frame_3dpose.tvec, scene_intrinsics)

                    # check whether box was propogated outside of the frame boundaries
                    if (curr_x <= 0) or (curr_y <= 0) or (curr_x >= (self.default_resolution[1] - 1)) or (curr_y >= (self.default_resolution[0] - 1)):
                        bbox, bbox_amodal, truncated = self._get_bbox_output(gt_box, trans_output, height, width)
                        if (bbox[2] - bbox[0]) <= 1 or (bbox[3] - bbox[1]) <= 1:
                            out_of_frame.add(ann['track_id'])
                            continue

                        self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids)
                        skipped_objs[ann['track_id']] = ann
                        continue

                    # compute new bounding box
                    new_box = np.array([curr_x - gt_box[2] / 2, curr_y - gt_box[3] / 2, gt_box[2], gt_box[3]])
                    ann['bbox'] = new_box

                    if flipped:
                        bbox = ann['bbox']
                        ann['bbox'] = [
                            width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

                    # truncate the box to frame boundaries
                    if ann['bbox'][0] < 0:
                        ann['bbox'][2] += ann['bbox'][0]
                        ann['bbox'][0] = 0

                    if ann['bbox'][1] < 0:
                        ann['bbox'][3] += ann['bbox'][1]
                        ann['bbox'][1] = 0

                    if (ann['bbox'][0] + ann['bbox'][2]) > self.opt.input_w:
                        ann['bbox'][2] += self.opt.input_w - (ann['bbox'][0] + ann['bbox'][2])

                    if (ann['bbox'][1] + ann['bbox'][3]) > self.opt.input_h:
                        ann['bbox'][3] += self.opt.input_h - (ann['bbox'][1] + ann['bbox'][3])

                    bbox, bbox_amodal, truncated = self._get_bbox_output(
                        ann['bbox'], trans_output, height, width)

                    if (bbox[2] - bbox[0]) <= 1 or (bbox[3] - bbox[1]) <= 1:
                        out_of_frame.add(ann['track_id'])
                        continue

                    box_size = ann['bbox'][2] * ann['bbox'][3]

                    if box_size < self.box_size_thresh[cls_id - 1] or last_seen_size < self.box_size_thresh[cls_id - 1]:
                        v = self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids)
                        skipped_objs[ann['track_id']] = ann
                        continue

                    is_added = self._add_instance(
                            ret, gt_det, counter, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
                            calib, pre_cts, track_ids, pre_ids, True)
                    if is_added:
                        counter += 1

                    ann['occl_length'] += 1
                    skipped_objs[ann['track_id']] = ann
                # otherwise use ground truth invisible annotation
                else:
                    if box_size < self.box_size_thresh[cls_id - 1]:
                        self._mask_ignore_or_crowd(ret, cls_id, bbox, ann['track_id'], pre_cts, track_ids)
                        continue
                    
                    if 'occl_length' not in ann:
                        ann['occl_length'] = 1

                    is_added = self._add_instance(
                            ret, gt_det, counter, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
                            calib, pre_cts, track_ids, pre_ids, True)
                    if is_added:
                        counter += 1
                    ann['occl_length'] += 1

            ret['frame_id'] = img_info['frame_id']
            ret['video_id'] = img_info['video_id']

            ret['gt_det'] = gt_det
            if opt.pre_hm:
                ret['pre_hm'] = pre_hm

            ret['gt_det'] = self._format_gt_det(ret['gt_det'])

            meta = {
            'calib': calib,
            'c':c,
            's':s,
            'height':height,
            'width': width,
            'trans_input':trans_input,
            'trans_output':trans_output,
            'inp_height':inp.shape[1],
            'inp_width':inp.shape[2],
            'out_height': inp.shape[1] // self.opt.down_ratio,
            'out_width': inp.shape[2] // self.opt.down_ratio,
            }
            ret['meta'] = meta
            ret['image_path'] = os.path.join(self.img_dir, img_info['file_name'])
            rets.append(ret)

            pre_anns = copy.deepcopy(anns[i])
            pre_invis_anns = copy.deepcopy(invis_anns[i])

        return rets

    def update_pre(self, pre_anns, skipped_objs):
        updated = []
        for ann in pre_anns:
            if ann['track_id'] in skipped_objs:
                ann = skipped_objs[ann['track_id']]
            updated.append(ann)

        return updated


    def _load_data(self, index, input_len):
        coco = self.coco
        img_dir = self.img_dir
        img_id = self.images[index]
        image_info = coco.loadImgs(ids=[img_id])[0]

        # find the video corresponding to the sampled frame
        video_identifier = image_info['video_id'] if not 'sensor_id' in image_info else str(image_info['video_id']) + '_' + str(image_info['sensor_id'])
        video_frames = self.video_to_images[video_identifier]
        frame_id = image_info['frame_id']

        frame_ind = self.video_to_image_map[video_identifier][img_id]
        stride = 1
        
        if self.min_frame_dist is None:
            min_frame_dist = self.opt.min_frame_dist
        else:
            min_frame_dist = self.min_frame_dist

        if self.max_frame_dist is None:
            max_frame_dist = self.opt.max_frame_dist
        else:
            max_frame_dist = self.max_frame_dist

        # randomly sample a temporal stride
        if 'train' in self.split and self.stride is None:
            stride = random.randint(min_frame_dist, max_frame_dist - 1)

        # select a clip with a given stride
        if frame_ind < (input_len * stride - 1):
            selected_images = [video_frames[frame_ind]] * input_len
        else:
            selected_images = video_frames[frame_ind - input_len * stride + 1: frame_ind + 1: stride]

        # random temporal flipping
        if not self.opt.no_temp_flip and 'train' in self.split and random.random() > 0.5:
            selected_images.reverse()

        # load frames and annotations for the clip
        imgs = []
        anns = []
        img_infos = []
        img_paths = []
        for image_info in selected_images:
            img, ann, img_info, img_path = self._load_image_anns(image_info['id'], coco, img_dir)
            imgs.append(img)
            anns.append(ann)
            img_infos.append(img_info)
            img_paths.append(img_path)

        return imgs, anns, img_infos, img_paths
