# Multi-Gait Dataset Preprocessing Code

## Dataset Overview

The Multi-Gait dataset is a comprehensive multimodal gait analysis dataset. It can be downloaded from the Zenodo repository: [https://doi.org/10.5281/zenodo.15660286](https://doi.org/10.5281/zenodo.15660286).

## Quick Start with Preprocessed Data

For users who want to start immediately, we provide preprocessed `.npy` datasets. These can be found in the `NpyDataset` directory within the downloaded repository.

## Preprocessing Code Structure

The `Scripts` directory contains all necessary codes for data preprocessing:

### 1. `.npy` Dataset Generation Code
- **Description**: Converts raw gait data (.mat) into numpy (.npy) format for easier handling and analysis.
- **Key Features**:
  - Processes data from mmWave radars and multi-view cameras
  - Generates various gait data types, gender and ID labels

### 2. SMPL Mesh Generation Scripts
- **Description**: Converts Kinect 3D skeleton keypoints into SMPL (Skinned Multi-Person Linear) human mesh models.
- **Key Features**:
  - Automatically transforms Kinect keypoint data into widely-used SMPL format
  - Enhances the diversity of gait data by adding 3D human mesh representations
  - Maintains high precision alignment with original keypoint data with alignment error of 0.6 mm
