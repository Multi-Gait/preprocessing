import os
import re
import sys
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
from meters import Meters
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from train import train
from transform import transform
from save import save_pic, save_params
from load import load
import torch
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
import time
import logging

import argparse
import json
from orientationLoss import OrientationLoss


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

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=cfg.MODEL.GENDER,
        model_root='smplpytorch/native/models')
    

    if args.dataset_name == 'mdmGen':
        from orientationLoss import OrientationLoss_SMPL
        loss_fn = OrientationLoss_SMPL()
        from load import MDMGen_dataset
        batchsize = 1024

        

    meters = Meters()
    
    dataset = MDMGen_dataset(os.path.join(cfg.DATASET.PATH), seq_len=15)
    data_loader = DataLoader(dataset, batch_size = batchsize, shuffle=False)
    result = [torch.empty(0)]*5
    person_id_all = torch.empty((0))
    for batch_idx, (target, person_id) in tqdm(enumerate(data_loader)):
        res = train(smpl_layer, loss_fn, target,
                    logger, writer, device,
                    args, cfg, meters)
        meters.update_avg(meters.min_loss, k=target.shape[0])
        meters.reset_early_stop()
        logger.info("avg_loss:{:.4f}".format(meters.avg))
        for i in range(len(result)-1):
            result[i] = torch.cat((result[i],res[i].detach().cpu()))
        # save_pic(res, smpl_layer, file, logger, args.dataset_name, target)
        result[4] = torch.cat((result[4], person_id.detach().cpu()))

        torch.cuda.empty_cache()
    
    
    result[0] = result[0].reshape((-1,15,result[0].shape[1]))
    result[1] = result[1].reshape((-1,15,result[1].shape[1]))
    result[2] = result[2].reshape((-1,15,result[2].shape[1],result[2].shape[2]))
    result[3] = result[3].reshape((-1,15,result[3].shape[1],result[3].shape[2]))
    result[4] = result[4].reshape((-1,15,result[4].shape[1]))[:,1,:] # (sample_num,1)

    
    save_params(result, cfg.DATASET.TARGET_PATH, logger, args.dataset_name)


    logger.info(
        "Fitting finished! Average loss:     {:.9f}".format(meters.avg))
