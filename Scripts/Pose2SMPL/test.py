#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:30:42 2024

@author: lrf

verfy that SMPL paramter
SMPL(shape, pose, trans) = verts 
"""
import os
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from fit.tools.smpl_velocity   import augment_smpl_verts_with_velocity

smpl_mat_path = "/home/lrf/Dropbox/mmWave/data/activity/July25/SMPL/01";
smpl_layer = SMPL_Layer(
             center_idx=0,
             gender='male',
             model_root='smplpytorch/native/models')
    
for root, dirs, files in os.walk(smpl_mat_path):
    if dirs != []:
        continue
    mat_files = os.listdir(root)
    mat_files.sort()
    for file in mat_files:
        file_path = os.path.join(root,file)
        data  = scipy.io.loadmat(file_path)
        shape_params = torch.from_numpy(data['shape_params'].astype(np.float32))        
        pose_params  = torch.from_numpy(data['pose_params'].astype(np.float32))
        smpl_verts = data['smpl_verts']

        with torch.no_grad():
            verts_model, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        
        # Create a figure with two subplots
        fig = plt.figure(figsize=(12, 6))
        
        # First subplot for the first point cloud
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(smpl_verts[:, 0], smpl_verts[:, 1], smpl_verts[:, 2], c='r', marker='o')
        ax1.set_title('Point Cloud 1')
        
        # Second subplot for the second point cloud
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(verts_model[:, 0], verts_model[:, 1], verts_model[:, 2], c='b', marker='o')
        ax2.set_title('Point Cloud 2')
        
        # Show the plot
        plt.show()