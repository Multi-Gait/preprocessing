import pickle
from queue import Queue
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D

def excute_draw_smpl_kinect_joints(smpl_joint_cart, kinect_joint_cart, smpl_vert):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    # smpl_orientation
    # ax.quiver(0,0,0,smpl_oritation[:,0],smpl_oritation[:,1],smpl_oritation[:,2])
    # kinect_orientation
    # kinect_orientation = np.cross((kinect_joint_cart[22,:]-kinect_joint_cart[1,:]).squeeze(),(kinect_joint_cart[18,:]-kinect_joint_cart[1,:]).squeeze())/
    #     /np.linalg.norm(np.cross((kinect_joint_cart[22,:]-kinect_joint_cart[1,:]).squeeze(),(kinect_joint_cart[18,:]-kinect_joint_cart[1,:]).squeeze()))
    # ax.quiver(0,0,0,kinect_orientation[0],kinect_orientation[1],kinect_orientation[2])
    # joint
    ax.scatter(smpl_joint_cart[:,0],smpl_joint_cart[:,1],smpl_joint_cart[:,2],c='r',label='smpl')
    ax.scatter(kinect_joint_cart[:,0],kinect_joint_cart[:,1],kinect_joint_cart[:,2],c='b',s=10 ,label='kinect')
    # 6890 verts
    ax.scatter(smpl_vert[:,0],smpl_vert[:,1],smpl_vert[:,2],s=1)
    # joint labels
    for i in range(len(smpl_joint_cart)):
        ax.text(smpl_joint_cart[i,0],smpl_joint_cart[i,1],smpl_joint_cart[i,2],i)
    for i in range(len(kinect_joint_cart)):
        ax.text(kinect_joint_cart[i,0],kinect_joint_cart[i,1],kinect_joint_cart[i,2],i)
    smpl_relation = {
        0: [3,1,2],
        1: [4],
        2: [5],
        3: [9],
        4: [7],
        5: [8],
        # 6: [-1],
        7: [10],
        8: [11],
        9: [12,13,14],
        10: [],
        11: [],
        12: [15],
        13: [16],
        14: [17],
        15: [],
        16: [18],
        17: [19],
        18: [20],
        19: [21],
        20: [22],
        21: [23],
        22: [],
        23: [],
    }
    joint_wait_queue = Queue(maxsize=24)
    joint_wait_queue.put(0)
    while not joint_wait_queue.empty():
        start_joint = joint_wait_queue.get()
        end_joint_list = smpl_relation[start_joint]
        if len(end_joint_list) == 0:
            continue
        else:
            for end_joint in end_joint_list:
                
                ax.plot3D(xs=[smpl_joint_cart[start_joint][0],smpl_joint_cart[end_joint][0]],ys=[smpl_joint_cart[start_joint][1],smpl_joint_cart[end_joint][1]],zs=[smpl_joint_cart[start_joint][2],smpl_joint_cart[end_joint][2]])
                joint_wait_queue.put(end_joint)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.gca().set_box_aspect((1,2,1))
    plt.show()

def excute_draw_smpl(smpl_joint_cart,smpl_vert, smpl_oritation):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    # smpl_orientation
    ax.quiver(0,0,0,smpl_oritation[:,0],smpl_oritation[:,1],smpl_oritation[:,2])
    # joint
    ax.scatter(smpl_joint_cart[:,0],smpl_joint_cart[:,1],smpl_joint_cart[:,2],c='r',label='smpl')
    # joint labels
    for i in range(len(smpl_joint_cart)):
        ax.text(smpl_joint_cart[i,0],smpl_joint_cart[i,1],smpl_joint_cart[i,2],i)
    # 6890 verts
    ax.scatter(smpl_vert[:,0],smpl_vert[:,1],smpl_vert[:,2],s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.gca().set_box_aspect((1,1,2))
    plt.show()

def excute_draw_6890_verts(hc_verts):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(hc_verts[:,0],hc_verts[:,1],hc_verts[:,2],c='r', label='verts')
    plt.show()

def draw_kinect_and_smpl(smpl_file_path):
    with open(smpl_file_path,'rb') as f:
        smpl_param = pickle.load(f)
        original_file = smpl_param['original_file']
        pose_params = smpl_param['pose_params']
        shape_params = smpl_param['shape_params']
        Jtr = np.array(smpl_param["Jtr"]).squeeze() # joint
        Jtr[:,[0,1,2]] = Jtr[:,[0,2,1]]
        Jtr[:,[1]] = -Jtr[:,[1]]
        verts = np.array(smpl_param["verts"]).squeeze() # 6890 verts
        verts[:,[0,1,2]] = verts[:,[0,2,1]]
        verts[:,[1]] = -verts[:,[1]]
        orientation = np.array(smpl_param["orientation"])
        orientation[:,[0,1,2]] = orientation[:,[0,2,1]]
        orientation[:,[1]] = - orientation[:,[1]]

        # print('original_file:',original_file)
        # print('smpl_joint:',Jtr)

    original_mat = scio.loadmat(original_file)
    kinect_key_point = original_mat['pc_xyz_key']
    # kinect_key_point[:,[1]] = - kinect_key_point[:,[1]]
    # kinect_key_point[:,[0,1,2]] = kinect_key_point[:,[0,2,1]]
    # kinect_key_point[0] = (kinect_key_point[0]+kinect_key_point[0])/2
    # kinect_key_point -= kinect_key_point[0] # 将0号点放在原点
    excute_draw_smpl_kinect_joints(Jtr, kinect_key_point, verts)

def draw_kinect_and_smpl_mat(smpl_file_path):
    smpl_param = scio.loadmat(smpl_file_path)
    pose_params = smpl_param['pose_params']
    shape_params = smpl_param['shape_params']
    Jtr = np.array(smpl_param["Jtr"]).squeeze() # joint
    Jtr[:,[0,1,2]] = Jtr[:,[0,2,1]]
    Jtr[:,[1]] = -Jtr[:,[1]]
    verts = np.array(smpl_param["verts"]).squeeze() # 6890 verts
    verts[:,[0,1,2]] = verts[:,[0,2,1]]
    verts[:,[1]] = -verts[:,[1]]

    kinect_key_point = scio.loadmat(smpl_file_path.replace('pose2smpl','walkOnSpotSample'))
    kinect_key_point = kinect_key_point['pc_xyz_key']

    excute_draw_smpl_kinect_joints(Jtr, kinect_key_point, verts)


if __name__ == '__main__':
    draw_kinect_and_smpl_mat("D:/data/action_recognition_CDJ/raw_data/pose2smpl/000/pc_ti_kinect_key_08.mat")