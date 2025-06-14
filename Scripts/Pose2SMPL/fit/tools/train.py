import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

from tqdm import tqdm
sys.path.append(os.getcwd())
from save import save_single_pic

from draw_kinect_and_smpl import excute_draw_smpl


def init(smpl_layer, target, device, cfg):
    params = {}
    params["pose_params"] = torch.zeros(target.shape[0], 72)
    # params["pose_params"][:,1] = -1*torch.pi
    params["shape_params"] = torch.zeros(target.shape[0], 10)
    params["scale"] = torch.ones([1])

    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    # print(target_orientation)
    params["scale"] = params["scale"].to(device)

    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)

    optim_params = [{'params': params["pose_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["shape_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["scale"], 'lr': cfg.TRAIN.LEARNING_RATE*10},]
    optimizer = optim.Adam(optim_params)

    index = {}
    smpl_index = []
    dataset_index = []
    for tp in cfg.DATASET.DATA_MAP:
        smpl_index.append(tp[0])
        dataset_index.append(tp[1])

    index["smpl_index"] = torch.tensor(smpl_index).to(device)
    index["dataset_index"] = torch.tensor(dataset_index).to(device)

    return smpl_layer, params, target, optimizer, index


def train(smpl_layer, loss_fn, target,
          logger, writer, device,
          args, cfg, meters):
    res = []
    smpl_layer, params, target, optimizer, index = init(smpl_layer, target, device, cfg)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]
    
    if cfg.TRAIN.OPTIMIZE_SCALE:
        with torch.no_grad():
            verts, Jtr= smpl_layer(pose_params, th_betas=shape_params, th_trans=target[:,0,:].squeeze())
            params["scale"]*=(torch.max(torch.abs(target))/torch.max(torch.abs(Jtr)))
        # excute_draw_smpl(Jtr.cpu().detach().numpy().squeeze(),verts.cpu().detach().numpy().squeeze(),body_orientation.cpu().detach().numpy())
    else:
        pass
    
    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params, th_trans=target[:,0,:].squeeze())
        # point to point loss
        # loss = F.smooth_l1_loss(scale*Jtr.index_select(1, index["smpl_index"]),
                                # target.index_select(1, index["dataset_index"]))
        # oritational loss
        loss, jtr_error = loss_fn(scale, index, Jtr,  target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== 新增部分：将shape_params设置为最后k帧的平均值 =====
        k = 5  # 你可以根据需要调整k值
        if shape_params.shape[0] > k:  # 确保有足够多的帧
            # 计算最后k帧的平均值 (1, 10)
            last_k_avg = shape_params[-k:].mean(dim=0, keepdim=True)
            # 将平均值复制到所有帧 (target.shape[0], 10)
            shape_params.data = last_k_avg.expand_as(shape_params).clone()
        # =================================================

        meters.update_early_stop(float(loss))
        if meters.update_res:
            res = [pose_params, shape_params, verts, Jtr]
        if meters.early_stop:
            logger.info("Early stop at epoch {} !".format(epoch))
            break

        if epoch % cfg.TRAIN.WRITE == 0 or epoch<10:
            # logger.info("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
            #         epoch, float(loss),float(scale)))
            # print("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
            #          epoch, float(loss),float(scale)))
            writer.add_scalar('loss', float(loss), epoch)
            writer.add_scalar('learning_rate', float(
                optimizer.state_dict()['param_groups'][0]['lr']), epoch)
            # save_single_pic(res,smpl_layer,epoch,logger,args.dataset_name,target)

    logger.info('Train ended, min_loss = {:.4f}'.format(
        float(meters.min_loss)))
    return res
