# Getting Started

This document provides tutorials to train and evaluate PermaTrack. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Benchmark evaluation

### PD

To test our pretrained model on the validation set of PD, download the [model](https://s3.console.aws.amazon.com/s3/object/tri-ml-public?region=us-east-1&prefix=github/permatrack/pd_17fr_21ep_vis.pth), copy it to `$PermaTrack_ROOT/models/`, and run

~~~
cd $PermaTrack_ROOT/src
python test.py tracking --exp_id pd --dataset pd_tracking --dataset_version val --track_thresh 0.4 --load_model ../models/pd_17fr_21ep_vis.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --stream_test
~~~

This will give a Track mAP of `66.96` if set up correctly. You can append `--debug 4` to the above command to visualize the predictions.

Please note that we are ignoring ground truth invisible object annotations in the validation set of PD (methods are not penalized for missing those boxes), but we are using them to filter out predictions which have a high overlap with a ground truth invisible box (to avoid conting such predictions as false positives; this was important for a fair evaluation before we introdiced the visibility head). As a result, the perfromance of our method with and without visiblity estimation described in the paper does not change much on PD. In the main experiments we did not use the visiblity estimation during evaluation on PD, but you can add it by appending `--visibility --visibility_thresh_eval 0.2` to the above command. The expected Track mAP is `66.78`.

### KITTI Tracking

To test the tracking performance on the validation set of KITTI with our pretrained model, download the [model](https://s3.console.aws.amazon.com/s3/object/tri-ml-public?region=us-east-1&prefix=github/permatrack/kitti_half_pd_5ep.pth), copy it to `$PermaTrack_ROOT/models/`, and run

~~~
python test.py tracking --exp_id kitti_half --dataset kitti_tracking --dataset_version val_half --track_thresh 0.4 --load_model ../models/kitti_half_pd_5ep.pth --is_recurrent --gru_filter_size 7  --num_gru_layers 1 --visibility --visibility_thresh_eval 0.2 --stream_test
~~~

The expected Track mAP is `70.53`. Here Track AP evluation also takes into account ignore regions in KITTI annotations (detections falling into these regions are not counted as false positives).

### MOT17

To test the tracking performance on the validation set of MOT17, download the [model](https://s3.console.aws.amazon.com/s3/object/tri-ml-public?region=us-east-1&prefix=github/permatrack/mot_half.pth), copy it to `$PermaTrack_ROOT/models/`, and run

~~~
python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../models/mot_half_13fr_5ep_occlasinvis.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility
~~~

The expected IDF1 is `68.2`.

To test with Track Rebirth, run

~~~
python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../models/mot_half_13fr_5ep_occlasinvis.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --max_age 32
~~~

The expected IDF1 is `71.9`.

To test with public detections, run

~~~
python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --track_thresh 0.4 --load_model ../models/mot_half_13fr_5ep_occlasinvis.pth --is_recurrent --gru_filter_size 7 --num_gru_layers 1 --visibility_thresh_eval 0.1 --stream_test --only_ped --ltrb_amodal --visibility --public_det --load_results ../data/mot17/results/val_half_det.json
~~~

The expected IDF1 is `67.0`.

### nuScenes

To test the tracking performance on the validation set of nuScenes, download the [model](https://s3.console.aws.amazon.com/s3/object/tri-ml-public?region=us-east-1&prefix=github/permatrack/nu_stage_3_17fr.pth), copy it to `$PermaTrack_ROOT/models/`, update `motmetrics` with

~~~
pip install motmetrics==1.1.3
~~~

then run

~~~
CUDA_VISIBLE_DEVICES=1 python test.py tracking,ddd --exp_id nuscenes_tracking  --dataset nuscenes_tracking --track_thresh 0.1 --resume --is_recurrent --gru_filter_size 7 --stream_test --load_model ../models/nu_stage_3_17fr.pth --visibility
~~~

The expected AMOTA is `10.9`.

## Training
We have packed all the training scripts in the [experiments](../experiments) folder.
Each model is trained on 8 Tesla V100 GPUs with 32GB of memory.
If the training is terminated before finishing, you can use the same command with `--resume` to resume training. It will found the latest model with the same `exp_id`.
All experiments rely on existing pretrained models, we provide the links to the corresponding models directly in the training scripts.
