import torch
import torch.nn as nn
import torch.nn.functional as F
class OrientationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, scale, index, pred_jtr,target_jtr):
        jtr_loss =  F.smooth_l1_loss(scale*pred_jtr.index_select(1, index["smpl_index"]),
                                target_jtr.index_select(1, index["dataset_index"]))
        # 骨盆朝向损失
        smpl_pelvis_ori = torch.cross(pred_jtr[:,2,:]-pred_jtr[:,0,:],pred_jtr[:,1,:]-pred_jtr[:,0,:])
        target_pelvis_ori = torch.cross(target_jtr[:,22,:]-target_jtr[:,1,:],target_jtr[:,18,:]-target_jtr[:,1,:])
        pelvis_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_pelvis_ori,target_pelvis_ori),dim=1)\
            /(torch.norm(smpl_pelvis_ori,dim=1) * torch.norm(target_pelvis_ori,dim=1))
        )
        # 骨盆平衡损失
        smpl_pelvis_balance = pred_jtr[:,2,:]-pred_jtr[:,1,:]
        target_pelvis_balance = target_jtr[:,22,:]-target_jtr[:,18,:]
        pelvis_balance_loss = torch.mean(
            torch.sum(torch.mul(smpl_pelvis_balance, target_pelvis_balance),dim=1)\
            /(torch.norm(smpl_pelvis_balance,dim=1)*torch.norm(target_pelvis_balance,dim=1))
        ) 
        # 胸部朝向损失
        smpl_chest_ori = torch.cross(pred_jtr[:,13,:]-pred_jtr[:,9,:],pred_jtr[:,14,:]-pred_jtr[:,9,:])
        target_chest_ori = torch.cross(target_jtr[:,3,:]-target_jtr[:,2,:],target_jtr[:,11,:]-target_jtr[:,2,:])
        chest_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_chest_ori,target_chest_ori),dim=1)\
            /(torch.norm(smpl_chest_ori,dim=1) * torch.norm(target_chest_ori,dim=1))
        )
        # 胸部平衡损失（左右锁骨位置）
        smpl_chest_balance = pred_jtr[:,14,:]-pred_jtr[:,13,:]
        target_chest_balance = target_jtr[:,11,:]-target_jtr[:,4,:]
        chest_balance_loss = torch.mean(
            torch.sum(torch.mul(smpl_chest_balance,target_chest_balance),dim=1)\
            /(torch.norm(smpl_chest_balance,dim=1)*torch.norm(target_chest_balance,dim=1))
        )
        # 脊椎向量损失
        smpl_mid_spine = pred_jtr[:,6,:]-pred_jtr[:,3,:]
        target_mid_spine = target_jtr[:,2,:]-target_jtr[:,1,:]
        mid_spine_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_mid_spine,target_mid_spine),dim=1)\
            /(torch.norm(smpl_mid_spine,dim=1)*torch.norm(target_mid_spine,dim=1))
        )
        # 大腿向量损失
        smpl_right_upper_leg = pred_jtr[:,5,:]-pred_jtr[:,2,:]
        target_right_upper_leg = target_jtr[:,23,:]-target_jtr[:,22,:]
        right_upper_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_upper_leg,target_right_upper_leg),dim=1)\
            /(torch.norm(smpl_right_upper_leg,dim=1)*torch.norm(target_right_upper_leg,dim=1))
        )
        smpl_left_upper_leg = pred_jtr[:,4,:]-pred_jtr[:,1,:]
        target_left_upper_leg = target_jtr[:,19,:]-target_jtr[:,18,:]
        left_upper_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_upper_leg,target_left_upper_leg),dim=1)\
            /(torch.norm(smpl_left_upper_leg,dim=1)*torch.norm(target_left_upper_leg,dim=1))
        )        
        # 小腿向量损失
        smpl_right_lower_leg = pred_jtr[:,8,:]-pred_jtr[:,5,:]
        target_right_lower_leg = target_jtr[:,24,:]-target_jtr[:,23,:]
        right_lower_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_lower_leg,target_right_lower_leg),dim=1)\
            /(torch.norm(smpl_right_lower_leg,dim=1)*torch.norm(target_right_lower_leg,dim=1))
        )
        smpl_left_lower_leg = pred_jtr[:,7,:]-pred_jtr[:,4,:]
        target_left_lower_leg = target_jtr[:,20,:]-target_jtr[:,19,:]
        left_lower_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_lower_leg,target_left_lower_leg),dim=1)\
            /(torch.norm(smpl_left_lower_leg,dim=1)*torch.norm(target_left_lower_leg,dim=1))
        ) 
        # 脚部向量损失
        smpl_right_foot = pred_jtr[:,11,:]-pred_jtr[:,8,:]
        target_right_foot = target_jtr[:,25,:]-target_jtr[:,24,:]
        right_foot_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_foot,target_right_foot),dim=1)\
            /(torch.norm(smpl_right_foot,dim=1)*torch.norm(target_right_foot,dim=1))
        )  
        smpl_left_foot = pred_jtr[:,10,:]-pred_jtr[:,7,:]
        target_left_foot = target_jtr[:,21,:]-target_jtr[:,20,:]
        left_foot_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_foot,target_left_foot),dim=1)\
            /(torch.norm(smpl_left_foot,dim=1)*torch.norm(target_left_foot,dim=1))
        )  
        # 大臂向量损失
        smpl_right_upper_arm = pred_jtr[:,19,:]-pred_jtr[:,17,:]
        target_right_upper_arm = target_jtr[:,13,:]-target_jtr[:,12,:]
        right_upper_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_upper_arm,target_right_upper_arm),dim=1)\
            /(torch.norm(smpl_right_upper_arm,dim=1)*torch.norm(target_right_upper_arm,dim=1))
        )
        smpl_left_upper_arm = pred_jtr[:,18,:]-pred_jtr[:,16,:]
        target_left_upper_arm = target_jtr[:,6,:]-target_jtr[:,5,:]
        left_upper_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_upper_arm,target_left_upper_arm),dim=1)\
            /(torch.norm(smpl_left_upper_arm,dim=1)*torch.norm(target_left_upper_arm,dim=1))
        )        
        # 小臂向量损失
        smpl_right_lower_arm = pred_jtr[:,21,:]-pred_jtr[:,19,:]
        target_right_lower_arm = target_jtr[:,14,:]-target_jtr[:,13,:]
        right_lower_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_lower_arm,target_right_lower_arm),dim=1)\
            /(torch.norm(smpl_right_lower_arm,dim=1)*torch.norm(target_right_lower_arm,dim=1))
        )
        smpl_left_lower_arm = pred_jtr[:,20,:]-pred_jtr[:,18,:]
        target_left_lower_arm = target_jtr[:,7,:]-target_jtr[:,6,:]
        left_lower_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_lower_arm,target_left_lower_arm),dim=1)\
            /(torch.norm(smpl_left_lower_arm,dim=1)*torch.norm(target_left_lower_arm,dim=1))
        )        

        loss = jtr_loss + (1-pelvis_ori_loss) + (1-chest_ori_loss) + (1-pelvis_balance_loss) + (1-chest_balance_loss)\
                + (1-left_upper_leg_ori_loss) + (1-left_lower_leg_ori_loss) + (1-right_upper_leg_ori_loss) + (1-right_lower_leg_ori_loss)\
                + (1-left_upper_arm_ori_loss) + (1-left_lower_arm_ori_loss) + (1-right_upper_arm_ori_loss) + (1-right_lower_arm_ori_loss)\
                + (1-left_foot_ori_loss) + (1-right_foot_ori_loss)\
                + (1-mid_spine_ori_loss)
        return loss, jtr_loss

class OrientationLoss_COCO(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, scale, index, pred_jtr,target_jtr):
        # jtr_loss =  F.smooth_l1_loss(scale*pred_jtr.index_select(1, index["smpl_index"]),
        #                         target_jtr.index_select(1, index["dataset_index"]))
        
        jtr_loss = F.smooth_l1_loss(scale*pred_jtr[:,0,:],0.5*(target_jtr[:,11,:]+target_jtr[:,12,:]))
        
        # 骨盆朝向损失
        smpl_pelvis_ori = torch.cross(pred_jtr[:,2,:]-pred_jtr[:,0,:],pred_jtr[:,1,:]-pred_jtr[:,0,:])
        target_pelvis_ori = torch.cross(target_jtr[:,12,:]-0.5*(target_jtr[:,5,:]+target_jtr[:,6,:]),target_jtr[:,11,:]-0.5*(target_jtr[:,5,:]+target_jtr[:,6,:]))
        pelvis_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_pelvis_ori,target_pelvis_ori),dim=1)\
            /(torch.norm(smpl_pelvis_ori,dim=1) * torch.norm(target_pelvis_ori,dim=1))
        )
        # 骨盆平衡损失
        smpl_pelvis_balance = pred_jtr[:,2,:]-pred_jtr[:,1,:]
        target_pelvis_balance = target_jtr[:,12,:]-target_jtr[:,11,:]
        pelvis_balance_loss = torch.mean(
            torch.sum(torch.mul(smpl_pelvis_balance, target_pelvis_balance),dim=1)\
            /(torch.norm(smpl_pelvis_balance,dim=1)*torch.norm(target_pelvis_balance,dim=1))
        ) 
        # 胸部朝向损失
        smpl_chest_ori = torch.cross(pred_jtr[:,13,:]-pred_jtr[:,9,:],pred_jtr[:,14,:]-pred_jtr[:,9,:])
        target_chest_ori = torch.cross(target_jtr[:,5,:]-0.5*(target_jtr[:,11,:]+target_jtr[:,12,:]),target_jtr[:,6,:]-0.5*(target_jtr[:,11,:]+target_jtr[:,12,:]))
        chest_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_chest_ori,target_chest_ori),dim=1)\
            /(torch.norm(smpl_chest_ori,dim=1) * torch.norm(target_chest_ori,dim=1))
        )
        # 胸部平衡损失（左右肩膀位置）
        smpl_chest_balance = pred_jtr[:,17,:]-pred_jtr[:,16,:]
        target_chest_balance = target_jtr[:,6,:]-target_jtr[:,5,:]
        chest_balance_loss = torch.mean(
            torch.sum(torch.mul(smpl_chest_balance,target_chest_balance),dim=1)\
            /(torch.norm(smpl_chest_balance,dim=1)*torch.norm(target_chest_balance,dim=1))
        )
        # 脊椎向量损失
        smpl_mid_spine = pred_jtr[:,6,:]-pred_jtr[:,3,:]
        target_mid_spine = 0.5*(target_jtr[:,6,:]+target_jtr[:,5,:]) - 0.5*(target_jtr[:,12,:]+target_jtr[:,11,:])
        mid_spine_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_mid_spine,target_mid_spine),dim=1)\
            /(torch.norm(smpl_mid_spine,dim=1)*torch.norm(target_mid_spine,dim=1))
        )
        # 大腿向量损失
        smpl_right_upper_leg = pred_jtr[:,5,:]-pred_jtr[:,2,:]
        target_right_upper_leg = target_jtr[:,14,:]-target_jtr[:,12,:]
        right_upper_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_upper_leg,target_right_upper_leg),dim=1)\
            /(torch.norm(smpl_right_upper_leg,dim=1)*torch.norm(target_right_upper_leg,dim=1))
        )
        smpl_left_upper_leg = pred_jtr[:,4,:]-pred_jtr[:,1,:]
        target_left_upper_leg = target_jtr[:,13,:]-target_jtr[:,11,:]
        left_upper_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_upper_leg,target_left_upper_leg),dim=1)\
            /(torch.norm(smpl_left_upper_leg,dim=1)*torch.norm(target_left_upper_leg,dim=1))
        )        
        # 小腿向量损失
        smpl_right_lower_leg = pred_jtr[:,8,:]-pred_jtr[:,5,:]
        target_right_lower_leg = target_jtr[:,16,:]-target_jtr[:,14,:]
        right_lower_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_lower_leg,target_right_lower_leg),dim=1)\
            /(torch.norm(smpl_right_lower_leg,dim=1)*torch.norm(target_right_lower_leg,dim=1))
        )
        smpl_left_lower_leg = pred_jtr[:,7,:]-pred_jtr[:,4,:]
        target_left_lower_leg = target_jtr[:,15,:]-target_jtr[:,13,:]
        left_lower_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_lower_leg,target_left_lower_leg),dim=1)\
            /(torch.norm(smpl_left_lower_leg,dim=1)*torch.norm(target_left_lower_leg,dim=1))
        ) 
        # 大臂向量损失
        smpl_right_upper_arm = pred_jtr[:,19,:]-pred_jtr[:,17,:]
        target_right_upper_arm = target_jtr[:,8,:]-target_jtr[:,6,:]
        right_upper_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_upper_arm,target_right_upper_arm),dim=1)\
            /(torch.norm(smpl_right_upper_arm,dim=1)*torch.norm(target_right_upper_arm,dim=1))
        )
        smpl_left_upper_arm = pred_jtr[:,18,:]-pred_jtr[:,16,:]
        target_left_upper_arm = target_jtr[:,7,:]-target_jtr[:,5,:]
        left_upper_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_upper_arm,target_left_upper_arm),dim=1)\
            /(torch.norm(smpl_left_upper_arm,dim=1)*torch.norm(target_left_upper_arm,dim=1))
        )        
        # 小臂向量损失
        smpl_right_lower_arm = pred_jtr[:,21,:]-pred_jtr[:,19,:]
        target_right_lower_arm = target_jtr[:,10,:]-target_jtr[:,8,:]
        right_lower_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_lower_arm,target_right_lower_arm),dim=1)\
            /(torch.norm(smpl_right_lower_arm,dim=1)*torch.norm(target_right_lower_arm,dim=1))
        )
        smpl_left_lower_arm = pred_jtr[:,20,:]-pred_jtr[:,18,:]
        target_left_lower_arm = target_jtr[:,9,:]-target_jtr[:,7,:]
        left_lower_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_lower_arm,target_left_lower_arm),dim=1)\
            /(torch.norm(smpl_left_lower_arm,dim=1)*torch.norm(target_left_lower_arm,dim=1))
        )
        
        # 脚部向量损失
        # smpl_right_foot = pred_jtr[:,11,:]-pred_jtr[:,8,:]
        # target_right_foot = target_pelvis_ori
        # right_foot_ori_loss = torch.mean(
        #     torch.sum(torch.mul(smpl_right_foot,target_right_foot),dim=1)\
        #     /(torch.norm(smpl_right_foot,dim=1)*torch.norm(target_right_foot,dim=1))
        # )  
        # smpl_left_foot = pred_jtr[:,10,:]-pred_jtr[:,7,:]
        # target_left_foot = target_pelvis_ori
        # left_foot_ori_loss = torch.mean(
        #     torch.sum(torch.mul(smpl_left_foot,target_left_foot),dim=1)\
        #     /(torch.norm(smpl_left_foot,dim=1)*torch.norm(target_left_foot,dim=1))
        # )  
        loss = jtr_loss + (1-pelvis_ori_loss) + (1-chest_ori_loss) + (1-pelvis_balance_loss) + (1-chest_balance_loss)\
                + (1-left_upper_leg_ori_loss) + (1-left_lower_leg_ori_loss) + (1-right_upper_leg_ori_loss) + (1-right_lower_leg_ori_loss)\
                + (1-left_upper_arm_ori_loss) + (1-left_lower_arm_ori_loss) + (1-right_upper_arm_ori_loss) + (1-right_lower_arm_ori_loss)\
                + (1-mid_spine_ori_loss)
        return loss, jtr_loss
        

class OrientationLoss_SMPL(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, scale, index, pred_jtr,target_jtr):
        jtr_loss =  F.smooth_l1_loss(scale*pred_jtr.index_select(1, index["smpl_index"]),
                                target_jtr.index_select(1, index["dataset_index"]))
        # 骨盆朝向损失
        smpl_pelvis_ori = torch.cross(pred_jtr[:,2,:]-pred_jtr[:,0,:],pred_jtr[:,1,:]-pred_jtr[:,0,:])
        target_pelvis_ori = torch.cross(target_jtr[:,2,:]-target_jtr[:,0,:],target_jtr[:,1,:]-target_jtr[:,0,:])
        pelvis_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_pelvis_ori,target_pelvis_ori),dim=1)\
            /(torch.norm(smpl_pelvis_ori,dim=1) * torch.norm(target_pelvis_ori,dim=1))
        )
        # 骨盆平衡损失
        smpl_pelvis_balance = pred_jtr[:,2,:]-pred_jtr[:,1,:]
        target_pelvis_balance = target_jtr[:,2,:]-target_jtr[:,1,:]
        pelvis_balance_loss = torch.mean(
            torch.sum(torch.mul(smpl_pelvis_balance, target_pelvis_balance),dim=1)\
            /(torch.norm(smpl_pelvis_balance,dim=1)*torch.norm(target_pelvis_balance,dim=1))
        ) 
        # 胸部朝向损失
        smpl_chest_ori = torch.cross(pred_jtr[:,13,:]-pred_jtr[:,9,:],pred_jtr[:,14,:]-pred_jtr[:,9,:])
        target_chest_ori = torch.cross(target_jtr[:,13,:]-target_jtr[:,9,:],target_jtr[:,14,:]-target_jtr[:,9,:])
        chest_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_chest_ori,target_chest_ori),dim=1)\
            /(torch.norm(smpl_chest_ori,dim=1) * torch.norm(target_chest_ori,dim=1))
        )
        # 胸部平衡损失（左右锁骨位置）
        smpl_chest_balance = pred_jtr[:,14,:]-pred_jtr[:,13,:]
        target_chest_balance = target_jtr[:,14,:]-target_jtr[:,13,:]
        chest_balance_loss = torch.mean(
            torch.sum(torch.mul(smpl_chest_balance,target_chest_balance),dim=1)\
            /(torch.norm(smpl_chest_balance,dim=1)*torch.norm(target_chest_balance,dim=1))
        )
        # 脊椎向量损失
        smpl_mid_spine = pred_jtr[:,6,:]-pred_jtr[:,3,:]
        target_mid_spine = target_jtr[:,6,:]-target_jtr[:,3,:]
        mid_spine_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_mid_spine,target_mid_spine),dim=1)\
            /(torch.norm(smpl_mid_spine,dim=1)*torch.norm(target_mid_spine,dim=1))
        )
        # 大腿向量损失
        smpl_right_upper_leg = pred_jtr[:,5,:]-pred_jtr[:,2,:]
        target_right_upper_leg = target_jtr[:,5,:]-target_jtr[:,2,:]
        right_upper_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_upper_leg,target_right_upper_leg),dim=1)\
            /(torch.norm(smpl_right_upper_leg,dim=1)*torch.norm(target_right_upper_leg,dim=1))
        )
        smpl_left_upper_leg = pred_jtr[:,4,:]-pred_jtr[:,1,:]
        target_left_upper_leg = target_jtr[:,4,:]-target_jtr[:,1,:]
        left_upper_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_upper_leg,target_left_upper_leg),dim=1)\
            /(torch.norm(smpl_left_upper_leg,dim=1)*torch.norm(target_left_upper_leg,dim=1))
        )        
        # 小腿向量损失
        smpl_right_lower_leg = pred_jtr[:,8,:]-pred_jtr[:,5,:]
        target_right_lower_leg = target_jtr[:,8,:]-target_jtr[:,5,:]
        right_lower_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_lower_leg,target_right_lower_leg),dim=1)\
            /(torch.norm(smpl_right_lower_leg,dim=1)*torch.norm(target_right_lower_leg,dim=1))
        )
        smpl_left_lower_leg = pred_jtr[:,7,:]-pred_jtr[:,4,:]
        target_left_lower_leg = target_jtr[:,7,:]-target_jtr[:,4,:]
        left_lower_leg_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_lower_leg,target_left_lower_leg),dim=1)\
            /(torch.norm(smpl_left_lower_leg,dim=1)*torch.norm(target_left_lower_leg,dim=1))
        ) 
        # 脚部向量损失
        smpl_right_foot = pred_jtr[:,11,:]-pred_jtr[:,8,:]
        target_right_foot = target_jtr[:,11,:]-target_jtr[:,8,:]
        right_foot_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_foot,target_right_foot),dim=1)\
            /(torch.norm(smpl_right_foot,dim=1)*torch.norm(target_right_foot,dim=1))
        )  
        smpl_left_foot = pred_jtr[:,10,:]-pred_jtr[:,7,:]
        target_left_foot = target_jtr[:,10,:]-target_jtr[:,7,:]
        left_foot_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_foot,target_left_foot),dim=1)\
            /(torch.norm(smpl_left_foot,dim=1)*torch.norm(target_left_foot,dim=1))
        )  
        # 大臂向量损失
        smpl_right_upper_arm = pred_jtr[:,19,:]-pred_jtr[:,17,:]
        target_right_upper_arm = target_jtr[:,19,:]-target_jtr[:,17,:]
        right_upper_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_upper_arm,target_right_upper_arm),dim=1)\
            /(torch.norm(smpl_right_upper_arm,dim=1)*torch.norm(target_right_upper_arm,dim=1))
        )
        smpl_left_upper_arm = pred_jtr[:,18,:]-pred_jtr[:,16,:]
        target_left_upper_arm = target_jtr[:,18,:]-target_jtr[:,16,:]
        left_upper_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_upper_arm,target_left_upper_arm),dim=1)\
            /(torch.norm(smpl_left_upper_arm,dim=1)*torch.norm(target_left_upper_arm,dim=1))
        )        
        # 小臂向量损失
        smpl_right_lower_arm = pred_jtr[:,21,:]-pred_jtr[:,19,:]
        target_right_lower_arm = target_jtr[:,21,:]-target_jtr[:,19,:]
        right_lower_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_right_lower_arm,target_right_lower_arm),dim=1)\
            /(torch.norm(smpl_right_lower_arm,dim=1)*torch.norm(target_right_lower_arm,dim=1))
        )
        smpl_left_lower_arm = pred_jtr[:,20,:]-pred_jtr[:,18,:]
        target_left_lower_arm = target_jtr[:,20,:]-target_jtr[:,18,:]
        left_lower_arm_ori_loss = torch.mean(
            torch.sum(torch.mul(smpl_left_lower_arm,target_left_lower_arm),dim=1)\
            /(torch.norm(smpl_left_lower_arm,dim=1)*torch.norm(target_left_lower_arm,dim=1))
        )        

        loss = jtr_loss + (1-pelvis_ori_loss) + (1-chest_ori_loss) + (1-pelvis_balance_loss) + (1-chest_balance_loss)\
                + (1-left_upper_leg_ori_loss) + (1-left_lower_leg_ori_loss) + (1-right_upper_leg_ori_loss) + (1-right_lower_leg_ori_loss)\
                + (1-left_upper_arm_ori_loss) + (1-left_lower_arm_ori_loss) + (1-right_upper_arm_ori_loss) + (1-right_lower_arm_ori_loss)\
                + (1-left_foot_ori_loss) + (1-right_foot_ori_loss)\
                + (1-mid_spine_ori_loss)
        return loss, jtr_loss
    
class OrientationLoss_Mocap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, scale, index, pred_jtr, target_jtr):
        jtr_loss =  F.smooth_l1_loss(scale*pred_jtr.index_select(1, index["smpl_index"]),
                                target_jtr.index_select(1, index["dataset_index"]))
        loss = jtr_loss
        return loss, jtr_loss