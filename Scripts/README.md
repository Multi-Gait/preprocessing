# Scripts Directory README

This directory contains scripts for preprocessing data related to the Multi-Gait dataset.

## Mat2Npy
The `Mat2Npy` sub - directory holds the `mat2npy.py` script. This script is used to convert data from MATLAB format (`.mat`) to NumPy format (`.npy`). It plays a crucial role in generating the `.npy` datasets that can be directly used for further analysis. 

## Pose2SMPL
The `Pose2SMPL` sub - directory contains scripts for converting Kinect skeleton keypoints to SMPL (Skinned Multi - Person Linear) format data. This includes generating SMPL skeleton keypoints and SMPL skin vertices from the original Kinect keypoint data, which helps in enriching the dataset with more advanced and standardized 3D human model representations for gait analysis.
 
