# MXNet Multi Person Pose Estimation
This is a MXNet version of Realtime_Multi-Person_Pose_Estimation, original code is here https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation 
and here https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation

## Introduction
Code repo for reproducing 2017 CVPR Oral paper using MXNet.  

## Require
1. [MXNet](http://mxnet.io)
2. pip install tensorflow mxboard
3. pip install pycocotools

## Evalute
- `python evaluate/evaluation.py` to evaluate the model on [images seperated by the original author](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose/blob/master/image_info_val2014_1k.txt)
- It should have `mAP 0.598` for the original rtpose, original repo have `mAP 0.577` because we do left and right flip for heatmap and PAF for the evaluation. 

### Pretrained Models & Performance on the dataset split by the original rtpose.

|   Reported on paper (VGG19)| mAP in this repo (VGG19)| Trained from scratch in this repo| 
|  :------:     | :---------: | :---------: |
|   0.577      | 0.598     |  **0.614** |


## Training
- `cd training; bash getData.sh` to obtain the COCO images in `dataset/COCO/images/`, keypoints annotations in `dataset/COCO/annotations/`
- Download the mask of the unlabeled person at [Dropbox](https://www.dropbox.com/s/bd9ty7b4fqd5ebf/mask.tar.gz?dl=0) in `training/datasets/COCO/mask`
- Download the official training format at [Dropbox](https://www.dropbox.com/s/0sj2q24hipiiq5t/COCO.json?dl=0) in `training/datasets/COCO.json`

then run:


`python train_model.py --gpu_ids 0 --lr=1 --wd=0.00001 --momentum=0.9 --log_key="lr_1_wd_0.0001_momentum_0.9"`


## Related repository
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

### Network Architecture
- testing architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/pose.png)

- training architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/training_structure.png)

## Contributions

All contributions are welcomed. If you encounter any issue (including examples of images where it fails) feel free to open an issue.

## Citation
Please cite the paper in your publications if it helps your research:    

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }
      
    @INPROCEEDINGS{8486591, 
    author={H. Wang and W. P. An and X. Wang and L. Fang and J. Yuan}, 
    booktitle={2018 IEEE International Conference on Multimedia and Expo (ICME)}, 
    title={Magnify-Net for Multi-Person 2D Pose Estimation}, 
    year={2018}, 
    volume={}, 
    number={}, 
    pages={1-6}, 
    month={July},}
