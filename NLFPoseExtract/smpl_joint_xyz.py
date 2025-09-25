import os
import pickle
import torch
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

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

def track_largest_person(V_list, max_miss=3, min_track_len=10):
    """
    Track整个序列中“最大的人物”, 带容错机制
    V_list: list of [X,J,3]
    max_miss: 最大容忍丢失帧数
    min_track_len: 最小轨迹长度, 小于该值返回None
    return: J [T_person, J,3] 该人的轨迹
    """
    if len(V_list) == 0:
        return None

    # Step1: 在第一帧找到最大的人
    first_nonempty = None
    for t, v in enumerate(V_list):
        if v.shape[0] > 0:
            first_nonempty = t
            break
    if first_nonempty is None:
        return None

    frame0 = V_list[first_nonempty]
    sizes = [compute_person_size(frame0[i]) for i in range(frame0.shape[0])]
    largest_idx = int(np.argmax(sizes))

    trajectory = [frame0[largest_idx]]
    prev_idx, prev_frame = largest_idx, frame0
    miss_cnt = 0  # 连续丢失帧计数

    # Step2: 往后追踪
    for t in range(first_nonempty + 1, len(V_list)):
        curr_frame = V_list[t]
        if curr_frame.shape[0] == 0:
            miss_cnt += 1
            if miss_cnt > max_miss:
                break  # 轨迹终止
            continue

        matches = match_people(prev_frame, curr_frame)
        if prev_idx in matches:
            curr_idx = matches[prev_idx]
            trajectory.append(curr_frame[curr_idx])
            prev_idx, prev_frame = curr_idx, curr_frame
            miss_cnt = 0  # 匹配成功，清零
        else:
            miss_cnt += 1
            if miss_cnt > max_miss:
                break  # 轨迹终止

    # Step3: 检查轨迹长度
    if len(trajectory) < min_track_len:
        return None

    return torch.stack(trajectory, dim=0)  # [T_person,J,3]

def compute_motion_speed(V):
  """
  V_list: list, 长度为T
    - 每个元素是 [X, 24, 3] 的tensor, X是人数
  返回:
    global_motion: 全局运动速度 (考虑平移)
  """
  # 对于多个人物，Track最大的那个人物，对于单个人物，同样可以兼容
  J = track_largest_person(V)
  if J is None:
    return None

  # ---------- Global Motion Speed (含平移) ----------
  diff = J[1:] - J[:-1]                    # [T-1,24,3]
  disp = torch.norm(diff, dim=-1)          # [T-1,24]
  motion_curve_global = disp.mean(dim=1)   # [T-1]
  global_motion = motion_curve_global.mean().item()

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


if __name__ == "__main__":
  smpl_root = '/data2/ywh/DataProcessNew/motionarray_nowat/smpl'
  smpl_files = sorted(os.listdir(smpl_root))
  results = {}

  for smpl in tqdm(smpl_files):
    print(smpl)
    with open(os.path.join(smpl_root, smpl), 'rb') as f:
        data = pickle.load(f)
    joint3d = data['pose']['joints3d_nonparam']
    global_motion_speed = compute_motion_speed(joint3d)
    global_motion_range = compute_motion_range(joint3d)

    if global_motion_speed is None or global_motion_range is None:
      continue
    print(global_motion_speed, global_motion_range)

    results[smpl.replace(".pkl", "")] = {}
    results[smpl.replace(".pkl", "")]['motion_speed'] = global_motion_speed
    results[smpl.replace(".pkl", "")]['motion_range'] = global_motion_range

  with open("/data2/ywh/DataProcessNew/motionarray_nowat/motion.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)