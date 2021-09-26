# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation or training, you will need to setup datasets.

### ParalleDomain (PD)

This is the synthetic dataset used to pre-train our model. You can download the images together with annotations under this [link](https://tri-ml-public.s3.amazonaws.com/datasets/pd_release.tar.gz). After dowloading, copy the contents into `$PermaTrack_ROOT/data/pd`.

### KITTI Tracking

We use KITTI Tracking to train and evaluate the system in the real world. Following prior work, we will only use the training set (and create a validation set from it) for developing this project.

- Download [images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip), and [annotations](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and unzip. Place or symlink the data as below:

  ~~~
  ${PermaTrack_ROOT}
  |-- data
  `-- |-- kitti_tracking
      `-- |-- data_tracking_image_2
          |   |-- training
          |   |-- |-- image_02
          |   |-- |-- |-- 0000
          |   |-- |-- |-- ...
          |-- |-- testing
          |-- label_02
          |   |-- 0000.txt
          |   |-- ...
  ~~~

- Run `python convert_kittitrack_to_coco.py` in `tools` to convert the annotation into COCO format. 
- The resulting data structure should look like:

  ~~~
  ${PermaTrack_ROOT}
  |-- data
  `-- |-- kitti_tracking
      `-- |-- data_tracking_image_2
          |   |-- training
          |   |   |-- image_02
          |   |   |   |-- 0000
          |   |   |   |-- ...
          |-- |-- testing
          |-- label_02
          |   |-- 0000.txt
          |   |-- ...
          |-- data_tracking_calib
          |-- label_02_val_half
          |   |-- 0000.txt
          |   |-- ...
          |-- label_02_train_half
          |   |-- 0000.txt
          |   |-- ...
          `-- annotations
              |-- tracking_train.json
              |-- tracking_test.json
              |-- tracking_train_half.json
              `-- tracking_val_half.json
  ~~~

To convert the annotation in a suitable format for evaluating track AP, run this command in `tools`: `python convert_kitti_to_tao.py` 

### MOT 2017

MOT is also used to train and evaluate the system. Again, we will only use the training set (and create a validation set from it) for developing this project.

We have reused the dataset preprocessing script from CenterTrack.

~~~
    cd $PermaTrack_ROOT/tools/
    bash get_mot_17.sh
~~~

The script includes:

- Download and unzip the dataset from [MOT17 website](https://motchallenge.net/data/MOT17/).
- Convert it into COCO format using `tools/convert_mot_to_coco.py`.
- Create the half-half train/ val set.
- Interpolate the training annotations over occlusions, as described in the paper.
- Convert the public detection into a specific format.
- The output data structure should be:

  ~~~
  ${PermaTrack_ROOT}
  |-- data
  `-- |-- mot17
      `-- |--- train
          |   |--- MOT17-02-FRCNN
          |   |    |--- img1
          |   |    |--- gt
          |   |    |   |--- gt.txt
          |   |    |   |--- gt_train_half.txt
          |   |    |   |--- gt_val_half.txt
          |   |    |--- det
          |   |    |   |--- det.txt
          |   |    |   |--- det_train_half.txt
          |   |    |   |--- det_val_half.txt
          |   |--- ...
          |--- test
          |   |--- MOT17-01-FRCNN
          |---|--- ...
          `---| annotations
              |--- train_half.json
              |--- train_half_interp.json
              |--- val_half.json
              |--- train.json
              |--- train_interp.json
              `--- test.json
  ~~~


### nuScenes

nuScenes is used for training and evaluating 3D object tracking. We also used nuScenes for pretraining KITTI models.


- Download the dataset from [nuScenes website](https://www.nuscenes.org/download). You only need to download the "Keyframe blobs", and only need the images data. You also need to download the maps and all metadata to make the nuScenes API happy.


- Unzip, rename, and place (or symlink) the data as below. You will need to merge folders from different zip files.

  ~~~
  ${PermaTrack_ROOT}
  |-- data
  `-- |-- nuscenes
      `-- |-- v1.0-trainval
          |   |-- samples
          |   |   |-- CAM_BACK
          |   |   |   | -- xxx.jpg
          |   |   |-- CAM_BACK_LEFT
          |   |   |-- CAM_BACK_RIGHT
          |   |   |-- CAM_FRONT
          |   |   |-- CAM_FRONT_LEFT
          |   |   |-- CAM_FRONT_RIGHT
          |-- |-- maps
          `-- |-- v1.0-trainval_meta
  ~~~

- Run `python convert_kitti_to_coco.py` in `tools` to convert the annotation into COCO format. It will create `train.json`, `val.json`, `test.json` under `data/nuscenes/annotations`. nuScenes API is required for running the data preprocessing.

### CrowdHuman

CrowdHuman is used for pretraining the MOT model. Only the training set is used.

- Download the dataset from [its website](https://www.crowdhuman.org/download.html).

- Unzip and place (or symlink) the data as below. You will need to merge folders from different zip files.

  ~~~
  ${PermaTrack_ROOT}
  |-- data
  `-- |-- crowdhuman
      |-- |-- CrowdHuman_train
      |   |   |-- Images
      |-- |-- CrowdHuman_val
      |   |   |-- Images
      |-- |-- annotation_train.odgt
      |-- |-- annotation_val.odgt
  ~~~

- Run `python convert_crowdhuman_to_coco.py` in `tools` to convert the annotation into COCO format. It will create `train.json`, `val.json` under `data/crowdhuman/annotations`.


## References
Please cite the corresponding References if you use the datasets.

~~~
  @article{MOT16,
    title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
    shorttitle = {MOT16},
    url = {http://arxiv.org/abs/1603.00831},
    journal = {arXiv:1603.00831 [cs]},
    author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
    month = mar,
    year = {2016},
    note = {arXiv: 1603.00831},
    keywords = {Computer Science - Computer Vision and Pattern Recognition}
  }

  @article{shao2018crowdhuman,
    title={Crowdhuman: A benchmark for detecting human in a crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv:1805.00123},
    year={2018}
  }

  @INPROCEEDINGS{Geiger2012CVPR,
    author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {CVPR},
    year = {2012}
  }

  @inproceedings{nuscenes2019,
  title={{nuScenes}: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
  booktitle={CVPR},
  year={2020}
  }
~~~
