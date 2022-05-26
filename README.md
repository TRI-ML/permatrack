# Learning to Track with Object Permanence
A video-based MOT approach capable of tracking through full occlusions:
![](readme/method.png)
> [**Learning to Track with Object Permanence**](https://arxiv.org/pdf/2103.14258.pdf),            
> Pavel Tokmakov, Jie Li, Wolfram Burgard, Adrien Gaidon,        
> *arXiv technical report ([arXiv 2103.14258](https://arxiv.org/pdf/2103.14258.pdf))*  


    @inproceedings{tokmakov2021learning,
      title={Learning to Track with Object Permanence},
      author={Tokmakov, Pavel and Li, Jie and Burgard, Wolfram and Gaidon, Adrien},
      booktitle={ICCV},
      year={2021}
    }

Check out our self-supervised extension publised at ICML'22:
> [**Object Permanence Emerges in a Random Walk along Memory**](https://arxiv.org/abs/2204.01784),    
> Pavel Tokmakov, Allan Jabri, Jie Li, Adrien Gaidon,   
> *arXiv technical report ([arXiv 2204.01784](https://arxiv.org/pdf/2204.01784.pdf))*


    @inproceedings{tokmakov2022object,
      title={Object Permanence Emerges in a Random Walk along Memory},
      author={Tokmakov, Pavel and Jabri, Allan and Li, Jie and Gaidon, Adrien},
      booktitle={ICML},
      year={2022}
    }

## Abstract
Tracking by detection, the dominant approach for online multi-object tracking, alternates between localization and association steps. As a result, it strongly depends on the quality of instantaneous observations, often failing when objects are not fully visible. In contrast, tracking in humans is underlined by the notion of object permanence: once an object is recognized, we are aware of its physical existence and can approximately localize it even under full occlusions. In this work, we introduce an end-to-end trainable approach for joint object detection and tracking that is capable of such reasoning. We build on top of the recent CenterTrack architecture, which takes pairs of frames as input, and extend it to videos of arbitrary length. To this end, we augment the model with a spatio-temporal, recurrent memory module, allowing it to reason about object locations and identities in the current frame using all the previous history. It is, however, not obvious how to train such an approach. We study this question on a new, large-scale, synthetic dataset for multi-object tracking, which provides ground truth annotations for invisible objects, and propose several approaches for supervising tracking behind occlusions. Our model, trained jointly on synthetic and real data, outperforms the state of the art on KITTI and MOT17 datasets thanks to its robustness to occlusions.

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.

## License

PermaTrack is developed upon [CenterTrack](https://github.com/xingyizhou/CenterTrack). Both codebases are released under MIT License themselves. Some code of CenterTrack are from third-parties with different licenses, please check the CenterTrack repo for details. In addition, this repo uses [py-motmetrics](https://github.com/cheind/py-motmetrics) for MOT evaluation, [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for nuScenes evaluation and preprocessing, and [TAO codebase](https://github.com/TAO-Dataset/tao) for computing Track AP. ConvGRU implementation is adopted from [this](https://github.com/happyjin/ConvGRU-pytorch) repo. See [NOTICE](NOTICE) for detail. Please note the licenses of each dataset. Most of the datasets we used in this project are under non-commercial licenses.

