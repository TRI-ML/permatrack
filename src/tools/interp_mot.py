import json
from collections import defaultdict
import pycocotools.coco as coco
import copy

SPLITS = ['train_half', 'train']

def interpolate(track, start_ann, start_ind, end_ann, end_ind, max_id, frames):
    print('interpolating from %d to %d' % (start_ind, end_ind))
    start_box = start_ann['bbox']
    end_box = end_ann['bbox']
    len_occl = end_ind - start_ind
    x_step = (end_box[0] - start_box[0]) / len_occl
    y_step = (end_box[1] - start_box[1]) / len_occl
    step = 1
    for i in range(start_ind + 1, end_ind):
        new_ann = copy.deepcopy(start_ann)
        new_ann['bbox'][0] += step * x_step
        new_ann['bbox'][1] += step * y_step
        new_ann['occlusion'] = 0.01
        new_ann['id'] = max_id
        new_ann['image_id'] = frames[i]['id']
        max_id += 1
        track[i] = new_ann
        step += 1

    return max_id

def process_video(frames, dataset, max_id):
    tracks = {}
    for i, frame in enumerate(frames):
        invis_count = 0
        occl_count = 0
        ann_ids = dataset.getAnnIds(imgIds=[frame['id']])
        anns = dataset.loadAnns(ids=ann_ids)
        for ann in anns:
            track_id = ann['track_id']
            ann['occlusion'] = 1
            if track_id not in tracks:
                tracks[track_id] = [None] * len(frames)
            tracks[track_id][i] = ann

    for track_id in tracks.keys():
        track = tracks[track_id]
        last_seen = None
        start_ind = None
        in_occl = False 
        for i, ann in enumerate(track):
            if ann is not None and in_occl:
                max_id = interpolate(track, last_seen, start_ind, ann, i, max_id, frames)
                in_occl = False

            if ann is not None:
                last_seen = ann
                start_ind = i
            if ann is None and last_seen is not None:
                in_occl = True

    annotations = []
    for track_id in tracks.keys():
        track = tracks[track_id]
        for i, ann in enumerate(track):
            if ann is not None:
                annotations.append(ann)         

    return annotations, max_id
    

if __name__ == '__main__':
    for split in SPLITS:
        data = json.load(open('../../data/mot17/annotations/%s.json' % split))
        coco_anns = coco.COCO('../../data/mot17/annotations/%s.json' % split)

        max_id = -1
        for ann in data['annotations']:
            if ann['id'] > max_id:
                max_id = ann['id']

        max_id += 1

        video_to_images = defaultdict(list)
        video_to_image_map = {}
        for image in coco_anns.dataset['images']:
            video_to_images[image['video_id']].append(image)

        for vid_id in video_to_images.keys():
            images = video_to_images[vid_id]
            images.sort(key=lambda x: x['frame_id'])
            video_to_images[vid_id] = images

        annotations = []
        for vid_id in video_to_images.keys():
            annotations_vid, max_id = process_video(video_to_images[vid_id], coco_anns, max_id)
            annotations.extend(annotations_vid)

        data['annotations'] = annotations

        json.dump(data, open('../../data/mot17/annotations/%s_interp.json' % split, 'w'))