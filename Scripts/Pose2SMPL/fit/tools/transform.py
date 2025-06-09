import numpy as np

rotate = {
    'HumanAct12': [1., -1., -1.],
    'CMU_Mocap': [1., 1., 1.], # 'CMU_Mocap': [0.05, 0.05, 0.05],
    'UTD_MHAD': [-1., 1., -1.],
    'Human3.6M': [-0.001, -0.001, 0.001],
    'NTU': [1., 1., -1.],
    'HAA4D': [1., -1., -1.],
    'Kinect': [1., 1., 1.],
    'mRI': [1., 1., 1.],
    'HumanML3D':[1., 1., 1.],
}


def transform(name, arr: np.ndarray):
    for i in range(arr.shape[0]):
        origin = arr[i][0].copy() # 0号关节点
        for j in range(arr.shape[1]):
            arr[i][j] -= origin # 将骨架移动到原点
            for k in range(3):
                arr[i][j][k] *= rotate[name][k] # xyz坐标变换
    return arr
