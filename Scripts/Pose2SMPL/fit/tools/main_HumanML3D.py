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
    
    if args.dataset_name == 'Kinect':
        from orientationLoss import OrientationLoss
        loss_fn = OrientationLoss()
    elif args.dataset_name == 'mRI':
        from orientationLoss import OrientationLoss_COCO
        loss_fn = OrientationLoss_COCO()
        from load import mRI_dataset
        batchsize = 512
    elif args.dataset_name == 'HumanML3D':
        from orientationLoss import OrientationLoss_SMPL
        loss_fn = OrientationLoss_SMPL()
        from load import HumanML3D_dataset
        batchsize = 512
        

    meters = Meters()
    file_num = 0
    
    logger.info(cfg.DATASET.PATH)
    for root, dirs, files in os.walk(cfg.DATASET.PATH):
        for file in sorted(files):
            pattern = re.compile(r'.*\.npy')
            if not re.match(pattern,file):
                continue
            if os.path.exists(os.path.join(root,file).replace('new_joints','pose2smpl')):
                logger.info(f'skip file:{root}, {file}')
                file_num += 1
                continue
            logger.info(f'walk into file:{root}, {file}')

            # if not 'baseball_swing' in file:
            #     continue
            file_num += 1
            logger.info(
                'Processing file: {}    [{} / {}]'.format(file, file_num, len(files)))
            # target = torch.from_numpy(transform(args.dataset_name, load(args.dataset_name,
            #                                                             os.path.join(root, file)))).float()
            # logger.info("target shape:{}".format(target.shape))

            # dataset = mRI_dataset(os.path.join(root, file))
            try:
                dataset = HumanML3D_dataset(os.path.join(root, file))
                data_loader = DataLoader(dataset, batch_size = batchsize, shuffle=False)
                result = [torch.empty(0)]*4
                for batch_idx, target in tqdm(enumerate(data_loader)):
                    res = train(smpl_layer, loss_fn, target,
                                logger, writer, device,
                                args, cfg, meters)
                    meters.update_avg(meters.min_loss, k=target.shape[0])
                    meters.reset_early_stop()
                    logger.info("avg_loss:{:.4f}".format(meters.avg))
                    for i in range(len(result)):
                        result[i] = torch.cat((result[i],res[i].detach().cpu()))
                    # save_pic(res, smpl_layer, file, logger, args.dataset_name, target)

                    torch.cuda.empty_cache()
                for i in range(len(result)):
                    print(result[i].shape)
                save_params(result, os.path.join(root, file), logger, args.dataset_name)
            except Exception as e:
                logger.info(e)
    logger.info(
        "Fitting finished! Average loss:     {:.9f}".format(meters.avg))
