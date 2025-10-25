## 筛选出去除突变情况下，每个数据集中，人体动作最大的一批数据（自动筛选，后续需要手动筛选一批）

import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.checkUtils import *
from collections import deque
import shutil
import torch
import yaml
from pose_draw.draw_pose_main import draw_pose_to_canvas_np, reshapePool
from extractUtils import check_single_human_requirements, check_multi_human_requirements, human_select
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, TimeoutError
from decord import VideoReader
from fractions import Fraction
import io
import gc
from PIL import Image
from multiprocessing import Process
import json
import jsonlines
from webdataset import TarWriter
import math
import glob
import pickle
import copy
import traceback
from NLFPoseExtract.smpl_joint_xyz import compute_motion_speed, compute_motion_range
from NLFPoseExtract.smpl_joint_change import get_most_abrupt_change
from VITPoseExtract.pipeline import VITPosePipeline
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy


def collect_nlf(data):
    uncollected_smpl_poses = [item['nlfpose'] for item in data]
    smpl_poses = [[] for _ in range(len(uncollected_smpl_poses))]
    for frame_idx in range(len(uncollected_smpl_poses)):
        for person_idx in range(len(uncollected_smpl_poses[frame_idx])):  # 每个人（每个bbox）只给出一个pose
            if len(uncollected_smpl_poses[frame_idx][person_idx]) > 0:    # 有返回的骨骼
                smpl_poses[frame_idx].append(uncollected_smpl_poses[frame_idx][person_idx][0].cpu())
            else:
                smpl_poses[frame_idx].append(torch.zeros((24, 3), dtype=torch.float32).cpu())  # 没有检测到人，就放一个全0的
    return smpl_poses

def process_fn_video(src, meta_dict=None):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue

        item = r.copy()
        meta = meta_dict.get(r['__key__'], None)
        if meta is None:
            print(f"skip {r['__key__']}, no meta")
            continue
        r.update(meta)
        ori_meta = meta.get('ori_meta', {})
        if isinstance(ori_meta, dict):
            r.update(ori_meta)
        
        motion_indices = r.get('motion_indices', None)
        if motion_indices is None:
            print(f"skip {r['__key__']}, no motion_indices")
            continue
        
        item.update({'motion_indices': motion_indices})

        yield item

def get_motion_dict(wds_path, save_dir_smpl):
    obj_list = []
    meta_dict = {}
    motion_score_dict = {}
    meta_file = wds_path.replace('.tar', '.meta.jsonl')
    meta_lines = open(meta_file).readlines()

    for meta_line in meta_lines:
        meta_line = meta_line.strip()
        try:
            meta = json.loads(meta_line)
        except Exception as e:
            print(e)
            print('json load error: ', meta_file)
            continue
        meta_dict[meta['key']] = meta
    dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_video, meta_dict=meta_dict),
        )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, collate_fn=lambda x: x[0])
    for data in tqdm(dataloader):
        key = data['__key__']
        motion_indices = data['motion_indices']
        try:
            smpl_path = os.path.join(save_dir_smpl, key + '.pkl')
            ori_smpl = pickle.load(open(smpl_path, 'rb'))
            smpl_ori_data = collect_nlf(ori_smpl)
            input_data = [torch.stack(smpl_ori_data[i]) for i in motion_indices]
            max_dz = get_most_abrupt_change(input_data)
            if max_dz > 1200:
                # print(f"skip {key}, max_dz too large: {max_dz}")
                continue
            else:
                motion_speed = compute_motion_speed(input_data)
                if motion_speed is not None and motion_speed > 12:
                    motion_score_dict[key] = motion_speed
        except Exception as e:
            print(f"skip {key}, smpl load error: {e}")
            print(f"Error traceback:")
            traceback.print_exc()
            continue
    return motion_score_dict
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_tar_list_to_new_tar(wds_list, top_keys, output_dir):
    obj_list = []
    sample_list = []
    shard_size = 100
    shard_id = 0
    output_pattern = "%06d"
    for _, wds_path in tqdm(enumerate(wds_list)):
        meta_dict = {}
        meta_file = wds_path.replace('.tar', '.meta.jsonl')
        meta_lines = open(meta_file).readlines()
        for meta_line in meta_lines:
            meta_line = meta_line.strip()
            try:
                meta = json.loads(meta_line)
            except Exception as e:
                print(e)
                print('json load error: ', meta_file)
                continue
            meta_dict[meta['key']] = meta
        dataset = wds.DataPipeline(
                wds.SimpleShardList(wds_path, seed=None),
                wds.tarfile_to_samples(),
                partial(process_fn_video, meta_dict=meta_dict),
            )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: x[0])
        for data in tqdm(dataloader):
            key = data['__key__']
            # motion_indices = data['motion_indices']
            if key in top_keys:
                data.pop('motion_indices', None)
                sample_list.append(data)
                obj_list.append(meta_dict.get(key, None))
                if len(sample_list) >= shard_size:
                    shard_file = os.path.join(output_dir, output_pattern % shard_id) + '.tar'
                    jsonl_file = os.path.join(output_dir, output_pattern % shard_id) + '.meta.jsonl'
                    if os.path.exists(shard_file):
                        print(f"remove existing shard file: {shard_file} because we will overwrite it")
                        os.remove(shard_file)
                    
                    with TarWriter(shard_file) as writer:
                        for sample in sample_list:
                            writer.write(sample)
                    with open(jsonl_file, 'w', encoding='utf-8') as outfile:
                        writer = jsonlines.Writer(outfile)
                        writer.write_all(obj_list)
                        writer.close()
                    sample_list = []
                    obj_list = []
                    shard_id += 1



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929_step5_1013',
                        help='Input root')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_finetune_wds_0929_step5_1013',
                        help='Output root')

    args = parser.parse_args()
    config = load_config(args.config)

    wds_root = config.get('wds_root', '')
    video_root = config.get('video_root', '')
    filter_args = config.get('filter_args', {})
    finetune_num = filter_args.get('finetune_num', 0)
    if finetune_num <= 0:
        exit("do not choose finetune from this dataset")
    

    save_dir_keypoints = os.path.join(video_root, 'keypoints')
    save_dir_bboxes = os.path.join(video_root, 'bboxes')
    save_dir_dwpose_mp4 = os.path.join(video_root, 'dwpose')
    save_dir_dwpose_reshape_mp4 = os.path.join(video_root, 'dwpose_reshape')
    save_dir_hands = os.path.join(video_root, 'hands')
    save_dir_faces = os.path.join(video_root, 'faces')
    save_dir_caption = os.path.join(video_root, 'caption')
    save_dir_caption_multi = os.path.join(video_root, 'caption_multi')
    save_dir_smpl = os.path.join(video_root, 'smpl')
    save_dir_smpl_render = os.path.join(video_root, 'smpl_render')
    save_dir_smpl_render_aug = os.path.join(video_root, 'smpl_render_aug')

    input_dir = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    output_dir = os.path.join(args.output_root, os.path.basename(os.path.normpath(video_root)))
    os.makedirs(output_dir, exist_ok=True)
    # Split wds_list into chunks
    input_tar_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True))
    all_motion_dict = {}
    for input_tar in input_tar_paths[:200]:       # 仅测试，记得去掉
        motion_dict = get_motion_dict(input_tar, save_dir_smpl)
        all_motion_dict.update(motion_dict)

    sorted_motion_items = sorted(all_motion_dict.items(), key=lambda x: x[1], reverse=True)
    top_count = min(finetune_num, len(sorted_motion_items))  # 防止总数少于1000
    top_items = sorted_motion_items[:top_count]
    top_keys = [item[0] for item in top_items]
    print(f"debug: selected top {top_count} samples for finetune")
    process_tar_list_to_new_tar(input_tar_paths, top_keys, output_dir)








