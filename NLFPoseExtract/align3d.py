from sympy import beta
import torch
import numpy as np
from scipy.optimize import minimize


def solve_new_camera_params_central(three_d_points, focal_length, imshape, new_2d_points):
    """
    通过最小化原始2D投影点和新的2D投影点之间的误差，求解新的相机参数。
    
    参数:
        three_d_points (torch.Tensor): N*3 的 3D 点
        focal_length (float): 原始相机的焦距
        imshape (tuple): 图像的尺寸，例如 [512, 896]
        original_2d_points (torch.Tensor): N*2 的原始2D投影点
        new_2d_points (torch.Tensor): N*2 的新的2D投影点
    
    返回:
        m, n, p, q: 新的相机内参中的参数
    """
    
    # 原始相机内参矩阵
    K_orig = np.array([
        [focal_length, 0, imshape[1] / 2],
        [0, focal_length, imshape[0] / 2],
        [0, 0, 1]
    ])

    # 目标函数：最小化原始投影点和新的投影点之间的误差
    def objective(params):
        m, n, p, q = params
        # 构建新的相机内参矩阵
        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * n, imshape[0] / 2 + q],
            [0, 0, 1]
        ])
        
        # 计算新的2D投影点
        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)
        
        # 计算原始2D投影点和新的投影点之间的误差
        # 第0个投影点特殊处理
        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 * 8 + error

    # 初始化参数 m, beta, p, q
    initial_params = [1.0, 1.0, 0.0, 0.0]  # 初始值

    # 使用最小二乘法求解 p, q)
    result = minimize(objective, initial_params, bounds=[(0.7, 1.4), (0.9, 1.1), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])])

    # 输出求解结果
    m, n, p, q = result.x
    print(f"debug: solved camera params m={m}, n={n}, p={p}, q={q}")

    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * n, imshape[0] / 2 + q],
        [0, 0, 1]
    ])


    return K_final


def solve_new_camera_params_down(three_d_points, focal_length, imshape, new_2d_points):
    """
    通过最小化原始2D投影点和新的2D投影点之间的误差，求解新的相机参数。
    
    参数:
        three_d_points (torch.Tensor): N*3 的 3D 点
        focal_length (float): 原始相机的焦距
        imshape (tuple): 图像的尺寸，例如 [512, 896]
        original_2d_points (torch.Tensor): N*2 的原始2D投影点
        new_2d_points (torch.Tensor): N*2 的新的2D投影点
    
    返回:
        m, n, p, q: 新的相机内参中的参数
    """
    
    # 原始相机内参矩阵
    K_orig = np.array([
        [focal_length, 0, imshape[1] / 2],
        [0, focal_length, imshape[0] / 2],
        [0, 0, 1]
    ])

    # 目标函数：最小化原始投影点和新的投影点之间的误差
    def objective(params):
        m, n, p, q = params
        # 构建新的相机内参矩阵
        K_new = np.array([
            [focal_length * m , 0, imshape[1] / 2 + p],
            [0, focal_length * n, imshape[0] / 2 + q],
            [0, 0, 1]
        ])
        
        # 计算新的2D投影点
        new_projections = []
        for point in three_d_points:
            X, Y, Z = point
            u = (K_new[0, 0] * X / Z) + K_new[0, 2]
            v = (K_new[1, 1] * Y / Z) + K_new[1, 2]
            new_projections.append([u, v])
        new_projections = np.array(new_projections)
        
        # 计算原始2D投影点和新的投影点之间的误差
        # 第0个投影点特殊处理
        error0 = np.sum((new_2d_points[:1] - new_projections[:1]) ** 2)
        error = np.sum((new_2d_points[1:] - new_projections[1:]) ** 2)
        return error0 + error * 4

    # 初始化参数 m, beta, p, q
    initial_params = [1.0, 1.0, 0.0, 0.0]  # 初始值

    # 使用最小二乘法求解 p, q)
    result = minimize(objective, initial_params, bounds=[(0.9, 1.2), (0.7, 1.4), (-imshape[1], imshape[1]), (-imshape[0], imshape[0])])

    # 输出求解结果
    m, n, p, q = result.x
    print(f"debug: solved camera params m={m}, n={n}, p={p}, q={q}")

    K_final = np.array([
        [focal_length * m, 0, imshape[1] / 2 + p],
        [0, focal_length * n, imshape[0] / 2 + q],
        [0, 0, 1]
    ])


    return K_final