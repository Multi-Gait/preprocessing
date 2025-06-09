## Change Log
6-9-2025. Update the code. 


## Fitting SMPL Parameters by 3D-pose Key-points

The repository provides a tool to fit **SMPL parameters** from **Kinect_key** in Multi-Gait datasets.

The SMPL human body layer for Pytorch is from the [smplpytorch](https://github.com/gulvarol/smplpytorch) repository.

<p align="center">
<img src="assets/fit.gif" width="350"/>
<img src="assets/gt.gif" width="350"/>
</p>

## Setup

### 1. Install packages
*  You will need to install the dependencies listed in [environment.yml](environment.yml):
  
    * `conda env update -f environment.yml` in an existing environment, or
    * `conda env create -f environment.yml`, for a new environment

### 2. Download and setup the SMPL pickle files
  * Download the pickles files 
    * Download the models from the [SMPL website](http://smpl.is.tue.mpg.de/) by choosing "SMPL for Python users". Note that you need to comply with the [SMPL model license](http://smpl.is.tue.mpg.de/license_model).
  * Extract and copy the `models` folder into the `smplpytorch/native/` folder (or set the `model_root` parameter accordingly).

### 3. Configure the directory
  *  Edit [main_seq_as_batch_v2.py](fit/tools/main_seq_as_batch_v2.py) to replace the SMPL pickle filepath with your local directory
    https://github.com/Multi-Gait/preprocessing/blob/main/Scripts/Pose2SMPL/fit/tools/main_seq_as_batch_v2.py#L106

  *  Edit [Kinect.json](fit/configs/Kinect.json) to configure PATH of the inputs (the .mat files without SMPL) and TARGET_PATH of the output (the .mat files that contain SMPL parameters)


https://github.com/Multi-Gait/preprocessing/blob/140b49340a08bbb0758172c59b7acb89a7520f0f/Scripts/Pose2SMPL/fit/configs/Kinect.json#L15

https://github.com/Multi-Gait/preprocessing/blob/140b49340a08bbb0758172c59b7acb89a7520f0f/Scripts/Pose2SMPL/fit/configs/Kinect.json#L16
   
## Quick Start
### Convert Kinect Skeleton into SMPL Data
 
calculate the velocity and save original data, SMPL paramter with velocity into .mat file
```
python fit/tools/main_seq_as_batch_v2.py
```

### Verify the SMPL parameters (shape and pose) can produce the smpl_verts using the following script.
```
python verfiy_smpl_params.py
```
