import os
import pickle
import torch
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from NLFPoseExtract.nlf_draw import p3d_to_p2d

def compute_person_size(joints):
    """
    joints: [J,3]
    定义人物大小: 3D关节的整体bbox直径
    """
    diff = joints.max(dim=0).values - joints.min(dim=0).values
    return diff.norm().item()

def match_people(frame1, frame2):
    """
    匹配两个相邻帧的多人，基于关键点 L2 距离
    frame1: [N1,J,3]
    frame2: [N2,J,3]
    return: 字典 {i: j}, 表示frame1中i号人匹配到frame2的j号人
    """
    if frame1.shape[0] == 0 or frame2.shape[0] == 0:
        return {}

    N1, J, _ = frame1.shape
    N2 = frame2.shape[0]

    cost = torch.cdist(frame1.reshape(N1, -1), frame2.reshape(N2, -1))  # [N1,N2]
    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
    return {i: j for i, j in zip(row_ind, col_ind)}


def track_largest_person(V_list, H=None, W=None, K=None, key_joints=[3, 6], max_miss=3, min_track_len=15):
    """
    追踪整个序列中“最大且关键点在屏幕内的人”
    """
    if len(V_list) == 0:
        return None

    first_nonempty = None
    largest_idx = None

    # ✅ Step 1: 找到第一帧中关键点在屏幕内的人
    for t, v in enumerate(V_list):
        if v.shape[0] == 0:
            continue
        # 对每个人判断是否有关键点在屏幕内
        valid_indices = []
        for i in range(v.shape[0]):
            if H is not None and W is not None and K is not None:
                indices_in = check_indices_in2d(v[i], H, W, K)
                if any(idx in indices_in for idx in key_joints):
                    valid_indices.append(i)
        if len(valid_indices) > 0:
            # 在这些人里选最大的那个
            sizes = [compute_person_size(v[i]) for i in valid_indices]
            largest_idx = valid_indices[int(np.argmax(sizes))]
            first_nonempty = t
            break

    if first_nonempty is None:
        return None

    # ✅ Step 2: 从这一帧往后追踪
    frame0 = V_list[first_nonempty]
    trajectory = [frame0[largest_idx]]
    prev_idx, prev_frame = largest_idx, frame0
    miss_cnt = 0

    for t in range(first_nonempty + 1, len(V_list)):
        curr_frame = V_list[t]
        if curr_frame.shape[0] == 0:
            miss_cnt += 1
            if miss_cnt > max_miss:
                break
            continue

        matches = match_people(prev_frame, curr_frame)
        if prev_idx not in matches:
            miss_cnt += 1
            if miss_cnt > max_miss:
                break
            continue

        curr_idx = matches[prev_idx]
        curr_person = curr_frame[curr_idx]

        # ✅ 检查关键点是否还在屏幕内
        if H is not None and W is not None and K is not None:
            indices_in = check_indices_in2d(curr_person, H, W, K)
            if not any(idx in indices_in for idx in key_joints):
                miss_cnt += 1
                if miss_cnt > max_miss:
                    break
                continue

        trajectory.append(curr_person)
        prev_idx, prev_frame = curr_idx, curr_frame
        miss_cnt = 0

    if len(trajectory) < min_track_len:
        return None

    return torch.stack(trajectory, dim=0)


# def track_largest_person(V_list, max_miss=3, min_track_len=15):
#     """
#     Track整个序列中“最大的人物”, 带容错机制
#     V_list: list of [X,J,3]
#     max_miss: 最大容忍丢失帧数
#     min_track_len: 最小轨迹长度, 小于该值返回None
#     return: J [T_person, J,3] 该人的轨迹
#     """
#     if len(V_list) == 0:
#         return None

#     # Step1: 在第一帧找到最大的人
#     first_nonempty = None
#     for t, v in enumerate(V_list):
#         if v.shape[0] > 0:
#             first_nonempty = t
#             break
#     if first_nonempty is None:
#         return None

#     frame0 = V_list[first_nonempty]
#     sizes = [compute_person_size(frame0[i]) for i in range(frame0.shape[0])]
#     largest_idx = int(np.argmax(sizes))

#     trajectory = [frame0[largest_idx]]
#     prev_idx, prev_frame = largest_idx, frame0
#     miss_cnt = 0  # 连续丢失帧计数

#     # Step2: 往后追踪
#     for t in range(first_nonempty + 1, len(V_list)):
#         curr_frame = V_list[t]
#         if curr_frame.shape[0] == 0:
#             miss_cnt += 1
#             if miss_cnt > max_miss:
#                 break  # 轨迹终止
#             continue

#         matches = match_people(prev_frame, curr_frame)
#         if prev_idx in matches:
#             curr_idx = matches[prev_idx]
#             trajectory.append(curr_frame[curr_idx])
#             prev_idx, prev_frame = curr_idx, curr_frame
#             miss_cnt = 0  # 匹配成功，清零
#         else:
#             miss_cnt += 1
#             if miss_cnt > max_miss:
#                 break  # 轨迹终止

#     # Step3: 检查轨迹长度
#     if len(trajectory) < min_track_len:
#         return None

#     return torch.stack(trajectory, dim=0)  # [T_person,J,3]


def check_indices_in2d(joints, H, W, K):
    """
    joints: [24, 3]
    返回在屏幕内的关节索引列表
    """
    joints_np = joints.cpu().numpy()
    joints_homo = joints_np.T  # shape: [3, 24]
    joints_proj = K @ joints_homo  # [3, 24]
    joints_2d = joints_proj[:2] / joints_proj[2]  # 归一化成像平面坐标
    # 检查是否在图像范围内
    x, y = joints_2d
    mask_x = (x >= 0) & (x < W)
    mask_y = (y >= 0) & (y < H)
    mask_z = joints_np[:, 2] > 0  # 深度为正（在相机前面）
    mask = mask_x & mask_y & mask_z
    indices_in = np.where(mask)[0].tolist()
    return indices_in


    

def compute_motion_speed(V, H, W, camera):
    """
    V_list: list, 长度为T
      - 每个元素是 [X, 24, 3] 的tensor, X是人数
    返回:
      global_motion: 全局运动速度 (考虑平移)
    """
    J = track_largest_person(V, H, W, camera)
    if J is None:
        return 0

    motion_curve = []

    for t in range(1, len(J)):
        prev = J[t - 1].clone()  # [24, 3]
        curr = J[t].clone()      # [24, 3]

        # ---------- 消除全局平移，以关节3为参考点 ----------
        prev_relative = prev - prev[3:4]  # [24, 3] - [1, 3] = [24, 3]
        curr_relative = curr - curr[3:4]  # [24, 3] - [1, 3] = [24, 3]

        # ---------- 2️⃣ 只计算屏幕内的点 ----------
        valid_indices_prev = check_indices_in2d(prev, H, W, camera)  # 已定义函数，返回在屏幕内的索引
        valid_indices_curr = check_indices_in2d(curr, H, W, camera)
        valid_indices = list(set(valid_indices_prev) & set(valid_indices_curr))

        if len(valid_indices) <= 3:
            frame_motion = 0.0
        else:
            diff = curr_relative[valid_indices] - prev_relative[valid_indices]  # [N,3]
            frame_motion = (torch.norm(diff, dim=-1).mean()).item()
        motion_curve.append(frame_motion)

    if len(motion_curve) == 0:
        return 0

    motion_curve = torch.tensor(motion_curve, dtype=torch.float32)

    # ---------- 4️⃣ 取连续20帧中动作幅度最大的片段 ----------
    window = 20
    T = len(motion_curve)
    if T > window:
        best_start = 0
        best_sum = -1e9
        for start in range(0, T - window + 1):
            window_sum = motion_curve[start:start + window].sum().item()
            if window_sum > best_sum:
                best_sum = window_sum
                best_start = start
        motion_curve = motion_curve[best_start:best_start + window]

    # ---------- 5️⃣ 计算平均速度 ----------
    global_motion = motion_curve.mean().item()
    return global_motion

def compute_motion_range(V):
  """
  V_list: list, 长度为T
    - 每个元素是 [X, 24, 3] 的tensor, X是人数
  返回:
    global_range: 全局运动幅度 (考虑平移)
  """
   # 对于多个人物，Track最大的那个人物，对于单个人物，同样可以兼容
  J = track_largest_person(V)
  if J is None:
    return None

  # ---------- Global Motion Range (含平移) ----------
  # 看每个关节在整个序列中的 max-min 范围
  diff_global = J.max(dim=0).values - J.min(dim=0).values  # [24,3]
  global_range = diff_global.norm(dim=-1).mean().item()

  return global_range


def collect_nlf_for_speed_select(data):   # 如果没检测到，不考虑非0的
    uncollected_smpl_poses = [item['nlfpose'] for item in data]
    smpl_poses = [[] for _ in range(len(uncollected_smpl_poses))]
    for frame_idx in range(len(uncollected_smpl_poses)):
        for person_idx in range(len(uncollected_smpl_poses[frame_idx])):  # 每个人（每个bbox）只给出一个pose
            if len(uncollected_smpl_poses[frame_idx][person_idx]) > 0:    # 有返回的骨骼
                smpl_poses[frame_idx].append(uncollected_smpl_poses[frame_idx][person_idx][0].cpu())
    return smpl_poses


