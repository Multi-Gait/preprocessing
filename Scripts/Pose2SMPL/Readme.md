## Fitting SMPL Parameters by 3D-pose Key-points

The repository provides a tool to fit **SMPL parameters** from **3D-pose** datasets that contain key-points of human body.

The SMPL human body layer for Pytorch is from the [smplpytorch](https://github.com/gulvarol/smplpytorch) repository.

<p align="center">
<img src="assets/fit.gif" width="350"/>
<img src="assets/gt.gif" width="350"/>
</p>

## Setup

### 1. Install  `smplpytorch` package
* **Run without installing:** You will need to install the dependencies listed in [environment.yml](environment.yml):
  
  * `conda env update -f environment.yml` in an existing environment, or
  * `conda env create -f environment.yml`, for a new `smplpytorch` environment
* **Install:** To import `SMPL_Layer` in another project with `from smplpytorch.pytorch.smpl_layer import SMPL_Layer` do one of the following.
  * Option 1: pip install. You can install `smplpytorch` from [PyPI](https://pypi.org/project/smplpytorch/). Additionally, you might need to install [chumpy](https://github.com/hassony2/chumpy.git).
    ``` bash
    pip install smplpytorch
    ```
  * Option 2: Download the source codes. This should automatically install the dependencies.
    ``` bash
    git clone https://github.com/gulvarol/smplpytorch.git
    cd smplpytorch
    pip install .
    ```
### 2. Download and setup the SMPL pickle files
  * Download the pickles files 
    * Option 1: Download the models from [Dropbox](https://www.dropbox.com/scl/fo/2nlgomwnv3ar6igcipras/AKdXv04OZ1Z9MwP0DtJgbvQ?rlkey=948dezmme1qv23zj11e3utush&st=t17iojsh&dl=0).
    * Option 2: Download the models from the [SMPL website](http://smpl.is.tue.mpg.de/) by choosing "SMPL for Python users". Note that you need to comply with the [SMPL model license](http://smpl.is.tue.mpg.de/license_model).
  * Extract and copy the `models` folder into the `smplpytorch/native/` folder (or set the `model_root` parameter accordingly).

### 3. Configure the directory
  *  Edit [main_seq_as_batch_v2.py](fit/tools/main_seq_as_batch_v2.py) to replace the SMPL pickle filepath with your local directory
    https://github.com/liux4189/mmWave-human-sensing/blob/bbf8fb75231d56667318f4652be3487d2bedb16d/preprocessing/Pose2SMPLv2/fit/tools/main_seq_as_batch_v2.py#L106

  *  Edit [Kinect.json](fit/configs/Kinect.json) to configure PATH of the inputs (the .mat files without SMPL) and TARGET_PATH of the output (the .mat files that contain SMPL parameters)
    https://github.com/liux4189/mmWave-human-sensing/blob/1d15980f333a90adce484bcc2da9186953587463/preprocessing/Pose2SMPLv2/fit/configs/Kinect.json#L15
    https://github.com/liux4189/mmWave-human-sensing/blob/1d15980f333a90adce484bcc2da9186953587463/preprocessing/Pose2SMPLv2/fit/configs/Kinect.json#L16
   
## Quick Start
### Convert Kinect Skeleton into SMPL Data
 
calculate the velocity and save original data, SMPL paramter with velocity into .mat file
```
python fit/tools/main_seq_as_batch_v2.py --dataset_name Kinect
```

The arg `--dataset_name Kinect` will guide to the config file `config/Kinect.json` in which the `file_path` of both source (Kinect) data and target data is set.

The Kinect data shoud be `.mat` files that obtain from `.bag` files using preprocess scripts like `rosbag2mat.m`. See *Data_preprocess/rosbag2mat.m* for more information.

### Verify the SMPL parameters (shape and pose) can produce the smpl_verts using the following script.
```
python verfiy_smpl_params.py
```
