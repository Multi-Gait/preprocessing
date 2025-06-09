from display_utils import display_model
from label import get_label
import sys
import os
import re
from tqdm import tqdm
import numpy as np
import pickle
import scipy.io as scio

sys.path.append(os.getcwd())


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pic(res, smpl_layer, file, logger, dataset_name, target):
    _, _, verts, Jtr ,_= res
    file_name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/{}/picture/{}".format(dataset_name, file_name)
    os.makedirs(fit_path,exist_ok=True)
    logger.info('Saving pictures at {}'.format(fit_path))
    for i in tqdm(range(Jtr.shape[0])):
        display_model(
            {'verts': verts.cpu().detach(),
             'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=os.path.join(fit_path+"/frame_{:0>4d}".format(i)),
            batch_idx=i,
            show=False,
            only_joint=True)
    logger.info('Pictures saved')



# save original parameter and SMPL paramters
def save_params_seq_as_batch_v2(dataset_name , pose_params, shape_params, smpl_joints_all, smpl_verts_all, dir, logger, cfg, original_skel=None):
    if original_skel is not None:
        original_skel = original_skel.cpu().detach().numpy()
    
    if dataset_name == 'Kinect_yuan':
        save_dir = dir.replace(cfg.DATASET.PATH,cfg.DATASET.TARGET_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_names = os.listdir(dir)
        file_names.sort()
        for i in range(len(file_names)):
            params = {}
            params["pose_params"] = pose_params[i]
            params["shape_params"] = shape_params[i]
            params["smpl_joints"] = smpl_joints_all[i]
            params["smpl_verts"] = smpl_verts_all[i]
            # attach the original data
            file_name = file_names[i]
            origin_file_path = os.path.join(dir,file_name)
            params["pc_xyz_key"]    = scio.loadmat(origin_file_path)['pc_xyz_key']
            params["pc_xyz_kinect"] = scio.loadmat(origin_file_path)['pc_xyz_kinect']
            params["pc_xyziv_ti_0"]   = scio.loadmat(origin_file_path)['pc_xyziv_ti_0']
            params["pc_xyziv_ti_1"]   = scio.loadmat(origin_file_path)['pc_xyziv_ti_1']
            params["pc_xyziv_ti_2"]   = scio.loadmat(origin_file_path)['pc_xyziv_ti_2']
            scio.savemat(os.path.join(save_dir,file_name),params)
    elif dataset_name == 'Kinect':
        save_dir = dir.replace(cfg.DATASET.PATH, cfg.DATASET.TARGET_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_names = os.listdir(dir)
        file_names.sort()
        for i in range(len(file_names)):
            params = {}
            params["pose_params"] = pose_params[i]
            params["shape_params"] = shape_params[i]
            params["smpl_joints"] = smpl_joints_all[i]
            params["smpl_verts"] = smpl_verts_all[i]
            # attach the original data
            file_name = file_names[i]
            origin_file_path = os.path.join(dir, file_name)
            params["kinect_key"] = scio.loadmat(origin_file_path)['kinect_key']
            params["pc_xyziv_ti"] = scio.loadmat(origin_file_path)['pc_xyziv_ti']
            params["rgb_front"] = scio.loadmat(origin_file_path)['rgb_front']
            params["rgb_side"] = scio.loadmat(origin_file_path)['rgb_side']
            params["pc_raw_iq"] = scio.loadmat(origin_file_path)['pc_raw_iq']


            scio.savemat(os.path.join(save_dir, file_name), params)



def save_single_pic(res, smpl_layer, epoch, logger, dataset_name, target):
    _, _, verts, Jtr = res
    fit_path = "fit/output/{}/picture".format(dataset_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving pictures at {}'.format(fit_path))
    display_model(
        {'verts': verts.cpu().detach(),
            'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        savepath=fit_path+"/epoch_{:0>4d}".format(epoch),
        batch_idx=60,
        show=False,
        only_joint=False)
    logger.info('Picture saved')