import scipy.io
import h5py
import numpy as np
import json
import os
import re
import torch
from torch.utils.data import Dataset

from transform import transform


class mRI_dataset(Dataset):
    def __init__(self, path):
        self.target = torch.from_numpy(transform('mRI', load('mRI',path))).float()
    def __getitem__(self, index):
        return self.target[index]
    def __len__(self):
        return self.target.shape[0]
    
class HumanML3D_dataset(Dataset):
    def __init__(self, path):
        self.target = torch.from_numpy(transform('HumanML3D', load('HumanML3D',path))).float()
    def __getitem__(self, index):
        return self.target[index]
    def __len__(self):
        return self.target.shape[0]
    
class MDMGen_dataset(Dataset):
    def __init__(self, path, seq_len) -> None:
        mat_data = scipy.io.loadmat(path)
        self.target = mat_data['motion'][:,:,:,0:seq_len].transpose([0,3,1,2])
        sample_num = self.target.shape[0]
        self.target = np.reshape(self.target, (sample_num*seq_len, 24, 3))
        self.person_id = np.repeat(mat_data['person_id'], seq_len, axis=1).T
    def __getitem__(self, index):
        return np.squeeze(self.target[index,:,:]), self.person_id[index]
    def __len__(self):
        return self.target.shape[0]

def load(name, path):
    if name == 'UTD_MHAD':
        arr = scipy.io.loadmat(path)['d_skel']
        new_arr = np.zeros([arr.shape[2], arr.shape[0], arr.shape[1]])
        for i in range(arr.shape[2]):
            for j in range(arr.shape[0]):
                for k in range(arr.shape[1]):
                    new_arr[i][j][k] = arr[j][k][i]
        return new_arr
    elif name == 'HumanAct12':
        return np.load(path, allow_pickle=True)
    elif name == "CMU_Mocap":
        return np.load(path, allow_pickle=True)
    elif name == "Human3.6M":
        return np.load(path, allow_pickle=True)[0::5] # down_sample
    elif name == "NTU":
        return np.load(path, allow_pickle=True)[0::2]
    elif name == "HAA4D":
        return np.load(path, allow_pickle=True)
    elif name == 'Kinect':
        arr = scipy.io.loadmat(path)['kinect_key']
        # arr[0,:] = (arr[0,:]+arr[1,:])/2 # 调整0点位置，减少与SMPL差异
        arr[:,[1]] = - arr[:,[1]] # 绕竖直方向转180度
        arr[:,[0,1,2]] = arr[:,[0,2,1]] # 将y轴作为竖直方向
        arr = np.expand_dims(arr,axis=0)
        return arr
    elif name == 'mRI':
        arr = scipy.io.loadmat(path)
        refined_gt_kps = arr['refined_gt_kps']
        gt_avail_frames = arr['gt_avail_frames'].squeeze()
        refined_gt_kps = refined_gt_kps[gt_avail_frames[0]:gt_avail_frames[1],:,:]
        refined_gt_kps = refined_gt_kps.transpose(0,2,1)
        return refined_gt_kps
    elif name == 'HumanML3D':
        arr = np.load(path)
        return arr
        

def load_seq_as_batch(name, dir, pattern=None):
    if name == 'Kinect':
        seq_arr = []
        mat_files = os.listdir(dir)
        mat_files.sort()
        for file in mat_files:
            if pattern == None:
                file_path = os.path.join(dir,file)
            else:
                if re.match(pattern, file):
                    file_path = os.path.join(dir,file)
                else:
                    continue
            # print(file_path)
            try:
                arr = scipy.io.loadmat(file_path)['kinect_key']
                #arr = scipy.io.loadmat(file_path)['xyz_key']
            except:
                f = h5py.File(file_path,'r')
                arr = np.array(f.get('kinect_key'))
                arr = arr.swapaxes(0,1)
            arr[0,:] = (arr[0,:]+arr[1,:])/2 # 调整0点位置，减少与SMPL差异
            arr[:,[1]] = - arr[:,[1]]
            arr[:,[0,1,2]] = arr[:,[0,2,1]] # 将y轴作为竖直方向
            arr = arr.tolist()
            seq_arr.append(arr)
        seq_arr = np.asarray(seq_arr)
        
        return seq_arr
    if name == 'CMU_Mocap':
        arr = scipy.io.loadmat(dir)['mocap_joints']
        arr = arr.transpose((2,0,1))
        return arr
        
        
