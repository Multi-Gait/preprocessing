import os
import sys
sys.path.append(os.getcwd())
from meters import Meters
# from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import smplx
from train import train
from transform import transform
from save import save_pic, save_params, save_params_seq_as_batch
from load import load,load_seq_as_batch
import torch
import numpy as np
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
import time
import logging
import re

import argparse
import json



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Fit SMPL')
    parser.add_argument('--exp', dest='exp',
                        help='Define exp name',
                        default=time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())), type=str)
    parser.add_argument('--dataset_name', '-n', dest='dataset_name',
                        help='select dataset',
                        default='', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path',
                        help='path of dataset',
                        default=None, type=str)
    args = parser.parse_args()
    return args


def get_config(args):
    config_path = 'fit/configs/{}.json'.format(args.dataset_name)
    with open(config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    if not args.dataset_path == None:
        cfg.DATASET.PATH = args.dataset_path
    return cfg


def set_device(USE_GPU):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(os.path.join(cur_path, 'tb'))

    return logger, writer


if __name__ == "__main__":
    args = parse_args()

    cur_path = os.path.join(os.getcwd(), 'exp', args.exp)
    assert not os.path.exists(cur_path), 'Duplicate exp name'
    os.makedirs(cur_path)

    cfg = get_config(args)
    json.dump(dict(cfg), open(os.path.join(cur_path, 'config.json'), 'w'))

    logger, writer = get_logger(cur_path)
    logger.info("Start print log")

    device = set_device(USE_GPU=cfg.USE_GPU)
    logger.info('using device: {}'.format(device))

    smpl_layer = smplx.create('./smplx/models/', model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         num_betas=10,
                         num_expression_coeffs=10,
                         ext='pkl',
                         age='adult')
    
    if args.dataset_name == 'Kinect':
        from orientationLoss import OrientationLoss
        loss_fn = OrientationLoss()
    elif args.dataset_name == 'mRI':
        from orientationLoss import OrientationLoss_COCO
        loss_fn = OrientationLoss_COCO()
    elif args.dataset_name == 'CMU_Mocap':
        from orientationLoss import OrientationLoss_Mocap
        loss_fn = OrientationLoss_Mocap()
    meters = Meters()
    
    if args.dataset_name == 'Kinect':
        dir_num = 0
        logger.info(cfg.DATASET.PATH)
        for root, dirs, files in os.walk(cfg.DATASET.PATH):
            # logger.info(f'walk into file:{root}, {file}')
            if dirs != []:
                continue
            target_root = root.replace(cfg.DATASET.PATH, cfg.DATASET.TARGET_PATH)
            if os.path.exists(target_root):
                continue 
            dir_num += 1
            logger.info(
                'Processing files in: {}    [{} / {}]'.format(root, dir_num, len(dirs)))
            target = torch.from_numpy(transform(args.dataset_name, load_seq_as_batch(args.dataset_name, root, pattern='pc_ti_kinect_key_[0-9]+.mat'))).float()
            logger.info("target shape:{}".format(target.shape))
            res = train(smpl_layer, loss_fn, target,
                        logger, writer, device,
                        args, cfg, meters)
            meters.update_avg(meters.min_loss, k=target.shape[0])
            meters.reset_early_stop()
            logger.info("avg_loss:{:.4f}".format(meters.avg))

            save_params_seq_as_batch(args.dataset_name, res, root, logger, cfg)
            # save_pic(res, smpl_layer, file, logger, args.dataset_name, target)

            torch.cuda.empty_cache()
        logger.info(
            "Fitting finished! Average loss:     {:.9f}".format(meters.avg))
    elif args.dataset_name == 'CMU_Mocap':
        logger.info(cfg.DATASET.PATH)
        file_num = 0
        for root, dirs, files in os.walk(cfg.DATASET.PATH):
            # logger.info(f'walk into file:{root}, {file}')
            if dirs != []:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                target_file_path = file_path.replace(cfg.DATASET.PATH, cfg.DATASET.TARGET_PATH)
                if os.path.exists(target_file_path):
                    continue 
                file_num += 1
                logger.info(
                    'Processing files in: {}    [{} / {}]'.format(file_path, file_num, len(files)))
                target = torch.from_numpy(transform(args.dataset_name, load_seq_as_batch(args.dataset_name, file_path))).float()
                logger.info("target shape:{}".format(target.shape))
                res = train(smpl_layer, loss_fn, target,
                            logger, writer, device,
                            args, cfg, meters)
                meters.update_avg(meters.min_loss, k=target.shape[0])
                meters.reset_early_stop()
                logger.info("avg_loss:{:.4f}".format(meters.avg))

                save_params_seq_as_batch(args.dataset_name, res,  target_file_path, logger, cfg, target)
                # save_pic(res, smpl_layer, file, logger, args.dataset_name, target)

                torch.cuda.empty_cache()
        logger.info(
            "Fitting finished! Average loss:     {:.9f}".format(meters.avg))
