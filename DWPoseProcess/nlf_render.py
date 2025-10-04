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
from NLFPoseExtract.nlf_render import render_nlf_as_images
from NLFPoseExtract.reshape_utils_3d import reshapePool3d
import traceback
import taichi as ti
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy

def process_fn_video(src, meta_dict=None):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue

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

        yield {'__key__': r['__key__'], 'motion_indices': motion_indices}

def render_to_smpl_path(wds_path, save_dir_smpl, save_dir_smpl_render, reshape_type, save_dir_caption, save_dir_caption_multi):
    meta_dict = {}
    meta_file = wds_path.replace('.tar', '.meta.jsonl')
    meta_lines = open(meta_file).readlines()
    tmp_dir = '/dev/shm/tmp'
    ti.reset()       # 清理taichi的缓存
    tarname = os.path.basename(wds_path).replace('.tar', '')    
    ti.init(arch=ti.cuda, offline_cache_file_path=f'{tmp_dir}/{tarname}')
    os.makedirs(save_dir_smpl_render, exist_ok=True)

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
        motion_indices = data['motion_indices']
        smpl_data = pickle.load(open(os.path.join(save_dir_smpl, key + '.pkl'), 'rb'))
        multi_person_path = os.path.join(save_dir_caption_multi, key + '.txt')
        single_person_path = os.path.join(save_dir_caption, key + '.txt')
        multi_person = False
        if os.path.exists(multi_person_path):
            # print(f"debug: {key} is multi person")
            multi_person = True
        elif os.path.exists(single_person_path):
            multi_person = False
        if multi_person:
            reshape_type = "low"
        out_path_smpl_render = os.path.join(save_dir_smpl_render, key + '.mp4')
        reshape_pool_3d = reshapePool3d(reshape_type=reshape_type)
        smpl_render_data = render_nlf_as_images(smpl_data, motion_indices, reshape_pool=reshape_pool_3d)
        mpy.ImageSequenceClip(smpl_render_data, fps=16).write_videofile(out_path_smpl_render)
        

def process_tar_chunk(chunk, save_dir_smpl, save_dir_smpl_render, reshape_type, save_dir_caption, save_dir_caption_multi):
    for wds_path in chunk:
        render_to_smpl_path(wds_path, save_dir_smpl, save_dir_smpl_render, reshape_type, save_dir_caption, save_dir_caption_multi)
        gc.collect()
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929_step3',
                        help='Input root')
    parser.add_argument('--max_processes', type=int, default=8,
                        help='Max processes')
    parser.add_argument('--current_process', type=int, default=0,
                        help='Current process')

    args = parser.parse_args()
    config = load_config(args.config)

    wds_root = config.get('wds_root', '')
    video_root = config.get('video_root', '')
    

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

    processes = []  # 存储进程的列表
    max_processes = args.max_processes

    input_dir = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    # Split wds_list into chunks
    input_tar_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True))
    filter_args = config.get('filter_args', {})
    reshape_type = filter_args.get('reshape', 'normal')
    print(f"reshape_type: {reshape_type} for data: {video_root}")

    current_process = args.current_process
    current_tar_paths = input_tar_paths[current_process::max_processes]
    if len(current_tar_paths) == 0:
        print("No chunks to process")
    process_tar_chunk(current_tar_paths, save_dir_smpl, save_dir_smpl_render, reshape_type, save_dir_caption, save_dir_caption_multi)







    
