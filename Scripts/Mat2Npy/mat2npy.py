# -*- coding: utf-8 -*-
"""
Author：Multi-Gait_ruilishi

日期：2025.06

input: mat frame files

output: npy dateset

"""
import os
import glob
import scipy.io as scio
import numpy as np

# transform 32 Kinect joints into 24 SMPL points
def keypointtrans(data):
    res=np.zeros([24,3])
    res[0,:]=data[0,:]
    res[1, :] = data[18, :]
    res[2, :] = data[22, :]
    res[3, :] = data[1, :]
    res[4, :] = data[19, :]
    res[5, :] = data[23, :]
    res[6, :] = data[2, :]
    res[7, :] = data[20, :]
    res[8, :] = data[24, :]
    res[9, :] = data[2, :]
    res[10, :] = data[21, :]
    res[11, :] = data[25, :]
    res[12, :] = data[3, :]
    res[13, :] = data[4, :]
    res[14, :] = data[11, :]
    res[15, :] = data[26, :]
    res[16, :] = data[5, :]
    res[17, :] = data[12, :]
    res[18, :] = data[6, :]
    res[19, :] = data[13, :]
    res[20, :] = data[7, :]
    res[21, :] = data[14, :]
    res[22, :] = data[9, :]
    res[23, :] = data[16, :]
    return res

#Unify the number of point clouds in each frame of the point cloud to facilitate batch processing operations.
def unify_pc_ti(pc_xyziv_ti):
    pc_frame_ti = np.zeros((64, 5), dtype=np.float32)
    pc_no_ti = pc_xyziv_ti.shape[0]
    if pc_no_ti < 64:
        fill_list = np.random.choice(64, size=pc_no_ti, replace=False)
        fill_set = set(fill_list)
        pc_frame_ti[fill_list] = pc_xyziv_ti
        dupl_list = [x for x in range(64) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no_ti, size=len(dupl_list), replace=True)
        pc_frame_ti[dupl_list] = pc_xyziv_ti[dupl_pc]
    else:
        pc_list = np.random.choice(pc_no_ti, size=64, replace=False)
        pc_frame_ti = pc_xyziv_ti[pc_list]
    return pc_frame_ti



### configuration
frames_per_sample = 20 # last N frame to keep in each sample
rootdir = "/media/srl/T7/dataset_0818/smpl11_224"# please modify it to the root directory of dataset

outdir = rootdir

list_all_ti = []  # TI mmwave point cloud
list_all_kinect_key = []  # Kinect key points (32 joints)
list_all_image=[]    #
list_label_all = []       # subject ID
list_all_gender = []
len_total = []
list_all_smpl_vert=[]
list_all_smpl_key=[]
list_all_pc_rdi =[]
list_all_image_side = []


dirlist=np.arange(1,35)
dirlist=np.char.mod('%d',dirlist)
gender_indices = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]#1:male;0:female


for i in range(0,len(dirlist)):
    subDir = os.path.join(rootdir, dirlist[i])#/Subject_01, ...34
    if not os.path.isdir(subDir):
        continue   
    print("### Enter mmWave subdir ", subDir)
    subDirlist = os.listdir(subDir)
    len_total.append(len(subDirlist))
    
    # iterate through each sample of the subject
    for j in range(0, len(subDirlist)):#Subject_i/sample_01，02，03...04
        path = os.path.join(subDir, subDirlist[j])
        if not os.path.isdir(path):
            continue
    
        matFilelist = glob.glob(os.path.join(path, "*.mat"))
        matFilelist.sort()

        list_person_onesample_ti = []
        list_person_onesample_kinect_key = []
        list_person_onsample_gender = []
        list_person_onsample_image = []
        list_person_onsample_verts = []
        list_person_onsample_smpl_joints = []
        list_person_onsample_pc_rdi = []
        list_person_onsample_image_side = []

        if len(matFilelist) < frames_per_sample:
            continue
        
        # interate through each frame
        for frame in range(0, len(matFilelist)):
            # load multimodel data from mat frame
            data = scio.loadmat(matFilelist[frame])
            pc_xyziv_ti = data['pc_xyziv_ti']
            pc_rdi = data['pc_raw_rdi']
            pc_xyz_kinect_key = data['kinect_key']
            imageFormatted = np.asarray(data['rgb_front'])
            img_side = np.asarray(data['rgb_side'])

            smpl_verts=data['smpl_verts']
            smpl_joints = data['smpl_joints']


            if pc_xyziv_ti.shape[0]==0:
                continue


            pc_frame_kinect_key = keypointtrans(pc_xyz_kinect_key) #24 joints×3
            pc_frame_ti = unify_pc_ti(pc_xyziv_ti) #64 mmwave points × 5


            list_person_onesample_ti.append(pc_frame_ti)
            list_person_onesample_kinect_key.append(pc_frame_kinect_key)
            list_person_onsample_image.append(imageFormatted)
            list_person_onsample_verts.append(smpl_verts)
            list_person_onsample_smpl_joints.append(smpl_joints)
            list_person_onsample_image_side.append(img_side)
            list_person_onsample_pc_rdi.append(pc_rdi)


        
        # keep the last N frames
        list_person_onesample_ti_segment_frames = list_person_onesample_ti[-frames_per_sample:]
        list_person_onesample_kinect_key_segment_frames = list_person_onesample_kinect_key[-frames_per_sample:]
        list_person_onsample_image_segment_frames=list_person_onsample_image[-frames_per_sample:]

        list_person_onsample_verts=list_person_onsample_verts[-frames_per_sample:]
        list_person_onsample_smpl_joints=list_person_onsample_smpl_joints[-frames_per_sample:]

        list_person_onsample_image_side=list_person_onsample_image_side[-frames_per_sample:]
        list_person_onsample_pc_rdi = list_person_onsample_pc_rdi[-frames_per_sample:]

        list_label_all.append(dirlist[i])
        list_all_gender.append(gender_indices[i])

        list_all_ti.append(list_person_onesample_ti_segment_frames)
        list_all_kinect_key.append(list_person_onesample_kinect_key_segment_frames)
        list_all_image.append(list_person_onsample_image_segment_frames)
        list_all_smpl_vert.append(list_person_onsample_verts)
        list_all_smpl_key.append(list_person_onsample_smpl_joints)
        list_all_image_side.append(list_person_onsample_image_side)
        list_all_pc_rdi.append(list_person_onsample_pc_rdi)



# save the npy
list_all_ti = np.asarray(list_all_ti)
list_all_kinect_key = np.asarray(list_all_kinect_key)
list_label_all = np.asarray(list_label_all)
list_all_image=np.asarray(list_all_image)
list_all_gender=np.asarray(list_all_gender)
list_all_smpl_vert=np.asarray(list_all_smpl_vert)
list_all_smpl_key=np.asarray(list_all_smpl_key)
list_all_image_side=np.asarray(list_all_image_side)
list_all_pc_rdi=np.asarray(list_all_pc_rdi)

print("Total num of sample: ", len(list_all_ti))
np.save(os.path.join(outdir, "list_label_all.npy"), list_label_all) #human id,1-34
np.save(os.path.join(outdir, "list_all_gender.npy"), list_all_gender)# gender,1:male,0:female

np.save(os.path.join(outdir, "list_all_ti.npy"), list_all_ti) #mmWave Radar point cloud
np.save(os.path.join(outdir, "list_all_pc_rdi.npy"), list_all_pc_rdi)#range-doppler heatmaps

np.save(os.path.join(outdir, "list_all_kinect_key.npy"), list_all_kinect_key)# kinect skeleton joints
np.save(os.path.join(outdir, "list_all_image.npy"), list_all_image) #font-view images
np.save(os.path.join(outdir, "list_all_image_side.npy"), list_all_image_side)#side-view images

np.save(os.path.join(outdir, "list_all_smpl_vert.npy"), list_all_smpl_vert)#smpl 6890*4 verties
np.save(os.path.join(outdir, "list_all_smpl_key.npy"), list_all_smpl_key) #smpl 24*3 joints
