

## Change Log
6-9-2025. Update the code. 


## Convert the mat dataset to npy format for machine learning validation


## Quick Start

### Configure the directory

 replace the dataset filepath with your local directory, the dataset can be load at [https://zenodo.org/](https://zenodo.org/)

https://github.com/Multi-Gait/preprocessing/blob/5358c504bf4500d941ef76ad2a03ef94e205dc79/Scripts/Mat2Npy/mat2npy.py#L66


### Run the script
 
This script iterates over the dataset and outputs the npy format dataset corresponding to the multimodal data

```
python fit/tools/main_seq_as_batch_v2.py
```

### Verify the dataset root
```
python mat2npy.py
```



