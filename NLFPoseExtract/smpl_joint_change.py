import os
import pickle
import torch
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import json

def read_jsonl_file(file_path):
  data_dict = {}
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      for line_num, line in enumerate(f, 1):
        line = line.strip()
        if line:
          try:
            json_obj = json.loads(line)
            if 'key' in json_obj and 'motion_indices' in json_obj:
              data_dict[json_obj['key']] = json_obj['motion_indices']
            else:
              print(f"警告: 文件 {file_path} 第{line_num}行缺少必要字段")
          except json.JSONDecodeError as e:
            print(f"警告: 文件 {file_path} 第{line_num}行JSON解析错误: {e}")
  except Exception as e:
    print(f"错误: 无法读取文件 {file_path}: {e}")
  
  return data_dict

def read_all_jsonl_files(folder_path):
    result_dict = {}
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
        return result_dict
    
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            print(f"正在处理文件: {filename}")
            
            file_data = read_jsonl_file(file_path)
            result_dict.update(file_data)
            
            print(f"从 {filename} 中读取了 {len(file_data)} 条数据")
    
    return result_dict

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

def compute_abrupt_change(V_list):
    """
    计算每帧在 x/y/z 方向的平均关键点变化量。
    要求：
      - 连续两帧都非空才能计算变化；
      - 空帧（检测缺失）直接跳过；
    
    参数：
      V_list: list，每个元素是 [N, 24, 3] 的tensor（N是人数）
    
    返回：
      diffs: list[dict]，每个元素形如 {"frame": t, "dx": val, "dy": val, "dz": val}
             表示从第 t 帧到第 t+1 帧的平均位移
    """
    diffs = []
    J = track_largest_person(V_list)
    if J is None:
        return None

    # 遍历帧间
    for t in range(len(J) - 1):
        f1, f2 = J[t], J[t + 1]
        if f1.numel() == 0 or f2.numel() == 0:
            continue  # 跳过空帧

        diff = f2 - f1  # [24,3]
        dx = diff[:, 0].abs().mean().item()
        dy = diff[:, 1].abs().mean().item()
        dz = diff[:, 2].abs().mean().item()

        diffs.append({
            "frame": t,
            "dx": dx,
            "dy": dy,
            "dz": dz
        })

    return diffs

def get_most_abrupt_change(V_list):
    diffs = compute_abrupt_change(V_list)
    try:
        if diffs is None or len(diffs) == 0:
            return 0
        dz_values = [diff["dz"] for diff in diffs]
        max_dz = max(dz_values) if dz_values else 0
    except Exception as e:
        print(f"计算most_abrupt_change时出错: {e}")
        max_dz = 0
    return max_dz


if __name__ == "__main__":
  meta_data = '/data2/ywh/pose_packed_wds_0923_step4/bili_dance_hengping_250328'
  motion_indices_dict = read_all_jsonl_files(meta_data)

  file_root = '/data2/ywh/DataProcessNew/bili_dance_hengping_250328/smpl_render'
  smpl_root = '/data2/ywh/DataProcessNew/bili_dance_hengping_250328/smpl'
  smpl_files = sorted(os.listdir(file_root))
  results = {}

  for smpl in tqdm(smpl_files):
    smpl = smpl.replace(".mp4", ".pkl")
    key = smpl.replace(".pkl", "")

    if key not in motion_indices_dict:
       continue
    with open(os.path.join(smpl_root, smpl), 'rb') as f:
        data = pickle.load(f)
    joint3d = data['pose']['joints3d_nonparam']
    joint3d = [joint3d[i] for i in motion_indices_dict[key]]

    diffs = compute_abrupt_change(joint3d)
    if diffs is None:
       continue
    for d in diffs:
        if d['dz'] > 900:
           results[key] = {}
           results[key]['dx'] = d['dx']
           results[key]['dy'] = d['dy']
           results[key]['dz'] = d['dz']
           break
    
    with open("/data2/ywh/DataProcessNew/bili_dance_hengping_250328/abrupt_change.json", 'w', encoding='utf-8') as f:
      json.dump(results, f, indent=4, ensure_ascii=False)