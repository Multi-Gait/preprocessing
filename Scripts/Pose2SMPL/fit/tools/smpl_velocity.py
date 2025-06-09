# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 23:13:56 2024

@author: Ruofeng Liu
"""
import numpy as np

def augment_smpl_verts_with_velocity(verts, Jtr, pc_xyz_key):
    # transform the coordinate from Kinect to ROS
    verts, Jtr = np.asarray(verts), np.asarray(Jtr)
    verts[:, :, 2] *= -1
    verts[:, :, [1, 2]] = verts[:, :, [2, 1]]
    Jtr[:, :, 2] *= -1
    Jtr[:, :, [1, 2]] = Jtr[:, :, [2, 1]]
    verts, Jtr = verts.tolist(), Jtr.tolist()

    # augment the data with the velocity
    smpl_verts_all = [];
    smpl_joints_all = [];

    # zero velocity for the first frame    
    kinect_0_joints0 = np.asarray([pc_xyz_key[0][0,0], -1*pc_xyz_key[0][0,2], pc_xyz_key[0][0,1]])
    smpl_0, joints_0 = verts[0] + kinect_0_joints0, Jtr[0] + kinect_0_joints0
    smpl_verts = np.column_stack((smpl_0, np.zeros(len(smpl_0))))
    smpl_verts_all.append(smpl_verts);
    smpl_joints_all.append(joints_0)
    
    # compute the velocity of the remaining frames
    for i in range(len(verts) - 1):
        smpl_1 = verts[i]
        smpl_2 = verts[i + 1]
        joints_1 = Jtr[i]
        joints_2 = Jtr[i + 1]
        
        kinect_0_joint1 = np.asarray([pc_xyz_key[i][0,0], -1*pc_xyz_key[i][0,2], pc_xyz_key[i][0,1]])
        kinect_0_joint2 = np.asarray([pc_xyz_key[i + 1][0,0], -1*pc_xyz_key[i + 1][0,2], pc_xyz_key[i + 1][0,1]])

        # Translate
        smpl_1 = smpl_1 + kinect_0_joint1
        smpl_2 = smpl_2 + kinect_0_joint2  
        joints_1 = joints_1 + kinect_0_joint1
        joints_2 = joints_2 + kinect_0_joint2

        # Calculate dist_radius and velocity_radius
        dist_radius = np.sum((smpl_2 - smpl_1)*smpl_2, axis=1) / np.sqrt(np.sum(smpl_2**2, axis=1))
        #dist_radius = np.dot((smpl_2 - smpl_1), smpl_2.T) / np.sqrt(np.sum(smpl_2**2, axis=1))
        velocity_radius = dist_radius / 0.1

        # Combine smpl_2 with velocity_radius
        smpl_verts = np.column_stack((smpl_2, velocity_radius))
        smpl_joints = joints_2
        smpl_verts_all.append(smpl_verts);
        smpl_joints_all.append(smpl_joints)
        
    return smpl_verts_all, smpl_joints_all


"""
T_camera_to_r0
T_camera_to_r1
T_camera_to_r2
4D calibration matrix from camera to radarX 

"""
def augment_smpl_verts_with_velocity_3radars(verts, Jtr, pc_xyz_key, T_camera_to_r0, T_camera_to_r1, T_camera_to_r2):
    # obtain the position of each radar in the camera system
    T_r0_to_camera = np.linalg.inv(T_camera_to_r0)
    T_r1_to_camera = np.linalg.inv(T_camera_to_r1)
    T_r2_to_camera = np.linalg.inv(T_camera_to_r2)

    # Position and orientation of radar in camera frame
    position_r0_in_camera = T_r0_to_camera[:3, 3]
    position_r1_in_camera = T_r1_to_camera[:3, 3]
    position_r2_in_camera = T_r2_to_camera[:3, 3]
                          
    # transform the coordinate from Kinect to ROS
    verts, Jtr = np.asarray(verts), np.asarray(Jtr)
    verts[:, :, 2] *= -1
    verts[:, :, [1, 2]] = verts[:, :, [2, 1]]
    Jtr[:, :, 2] *= -1
    Jtr[:, :, [1, 2]] = Jtr[:, :, [2, 1]]
    verts, Jtr = verts.tolist(), Jtr.tolist()

    # augment the data with the velocity
    smpl_verts_all = [];
    smpl_joints_all = [];

    # zero velocity for the first frame    
    kinect_0_joints0 = np.asarray([pc_xyz_key[0][0,0], -1*pc_xyz_key[0][0,2], pc_xyz_key[0][0,1]])
    smpl_0, joints_0 = verts[0] + kinect_0_joints0, Jtr[0] + kinect_0_joints0
    smpl_verts = np.column_stack((smpl_0, np.zeros((len(smpl_0),3))))
    smpl_verts_all.append(smpl_verts);
    smpl_joints_all.append(joints_0)
    
    # compute the velocity of the remaining frames
    for i in range(len(verts) - 1):
        smpl_1 = verts[i]
        smpl_2 = verts[i + 1]
        joints_1 = Jtr[i]
        joints_2 = Jtr[i + 1]
        
        kinect_0_joint1 = np.asarray([pc_xyz_key[i][0,0], -1*pc_xyz_key[i][0,2], pc_xyz_key[i][0,1]])
        kinect_0_joint2 = np.asarray([pc_xyz_key[i + 1][0,0], -1*pc_xyz_key[i + 1][0,2], pc_xyz_key[i + 1][0,1]])

        # Translate
        smpl_1 = smpl_1 + kinect_0_joint1
        smpl_2 = smpl_2 + kinect_0_joint2  
        joints_1 = joints_1 + kinect_0_joint1
        joints_2 = joints_2 + kinect_0_joint2

        # Calculate dist_radius and velocity_radius
        radius_direction_r0 = (smpl_2-position_r0_in_camera)
        dist_radius_r0 = np.sum((smpl_2 - smpl_1)*radius_direction_r0, axis=1) / np.sqrt(np.sum(radius_direction_r0**2, axis=1))
        velocity_radius_r0 = dist_radius_r0 / 0.1

        radius_direction_r1 = (smpl_2-position_r1_in_camera)
        dist_radius_r1 = np.sum((smpl_2 - smpl_1)*radius_direction_r1, axis=1) / np.sqrt(np.sum(radius_direction_r1**2, axis=1))
        velocity_radius_r1 = dist_radius_r1 / 0.1
        
        radius_direction_r2 = (smpl_2-position_r2_in_camera)
        dist_radius_r2 = np.sum((smpl_2 - smpl_1)*radius_direction_r2, axis=1) / np.sqrt(np.sum(radius_direction_r2**2, axis=1))
        velocity_radius_r2 = dist_radius_r2 / 0.1

        # Combine smpl_2 with velocity_radius
        smpl_verts = np.column_stack((smpl_2, velocity_radius_r0, velocity_radius_r1, velocity_radius_r2))
        smpl_joints = joints_2
        smpl_verts_all.append(smpl_verts);
        smpl_joints_all.append(smpl_joints)
        
    return smpl_verts_all, smpl_joints_all