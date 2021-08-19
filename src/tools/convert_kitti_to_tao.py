import json


DATA_PATH = '../../data/kitti_tracking/'
SPLITS = ['tracking_val_half']


def get_cats_for_vid(anns, vid_id, image2vid):
    all_categories = set([])
    for cat in anns['categories']:
        all_categories.add(cat['id'])

    positive = set([])
    for ann in anns['annotations']:
        ann_vid_id = image2vid[ann['image_id']]
        if ann_vid_id == vid_id:
            category = ann['category_id']
            positive.add(category)

    return positive, all_categories - positive


def get_image2video_map(anns):
    mapping = {}
    for img in anns['images']:
        mapping[img['id']] = img['video_id']
        img['frame_index'] = img['frame_id'] - 1
    
    return mapping, anns


def unique_track_ids(anns, image2vid): 
    unique_tracks = {}
    track_counter = 0
    for ann in anns['annotations']:
        orig_track_id = ann['track_id']
        image_id = ann['image_id']
        ann['area'] = ann['bbox'][2] * ann['bbox'][3]
        video_id = image2vid[image_id]
        vid_track_pair = f"{video_id}_{orig_track_id}"
        if vid_track_pair not in unique_tracks:
            unique_tracks[vid_track_pair] = track_counter
            track_counter += 1
    
    tracks = []
    processed_tracks = set([])
    for ann in anns['annotations']:
        orig_track_id = ann['track_id']
        image_id = ann['image_id']
        video_id = image2vid[image_id]
        vid_track_pair = f"{video_id}_{orig_track_id}"
        ann['track_id'] = unique_tracks[vid_track_pair]
        if ann['track_id'] not in processed_tracks:
            track = {'id': ann['track_id'], 'category_id': ann['category_id'], 'video_id': video_id}
            processed_tracks.add(ann['track_id'])
            tracks.append(track)

    anns['tracks'] = tracks

    return anns



if __name__ == '__main__':
    ann_dir = DATA_PATH + '/annotations/'

    for split in SPLITS:
        print("Processing split %s" % split)
        anns = json.load(open(ann_dir + split + ".json"))
        image2vid, anns = get_image2video_map(anns)
        for vid in anns['videos']:
            print("Processing video %s" % vid['file_name'])
            vid['not_exhaustive_category_ids'] = []
            positives, negatives = get_cats_for_vid(anns, vid['id'], image2vid)
            vid['neg_category_ids'] = list(negatives)

        anns = unique_track_ids(anns, image2vid)

        json.dump(anns, open(ann_dir + split + "_tao.json", 'w'))