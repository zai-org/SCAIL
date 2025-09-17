import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.AAUtils import save_videos_from_pil
from DWPoseProcess.checkUtils import *
from collections import deque
import shutil
import torch
import yaml
from pose_draw.draw_pose_main import draw_pose_to_canvas, reshapePool
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

def process_fn_reshape(src):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue

        yield r

def reshape_dwpose_to_wds(wds_path, output_wds_path, save_dir_keypoints, save_dir_dwpose_mp4):
    meta_file = wds_path.replace('.tar', '.meta.jsonl')
    output_meta_file = output_wds_path.replace('.tar', '.meta.jsonl')
    dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_reshape),
        )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, collate_fn=lambda x: x[0])
    with TarWriter(output_wds_path) as writer:
        for data in tqdm(dataloader):
            key = data['__key__']
            mp4_bytes = data['mp4']
            vr = VideoReader(io.BytesIO(mp4_bytes))   # h w c
            height = vr[0].shape[0]
            width = vr[0].shape[1]
            keypoints_path = os.path.join(save_dir_keypoints, key + '.pt')
            out_path_dwpose_mp4 = os.path.join(save_dir_dwpose_mp4, key + '.mp4')
            keypoints = torch.load(keypoints_path, weights_only=False)
            reshape_scale = 0.6
            pool = reshapePool(alpha=reshape_scale)
            mp4_results = draw_pose_to_canvas(keypoints, pool=pool, H=height, W=width, reshape_scale=reshape_scale, points_only_flag=False, show_feet_flag=False)
            save_videos_from_pil(mp4_results, out_path_dwpose_mp4, fps=16) # 实际可能fps不一致，需要视频fps存，这个是否会影响训练？
            with open(out_path_dwpose_mp4, "rb") as f:
                dwpose_mp4_data = f.read()
            data['dwpose_reshape'] = dwpose_mp4_data
            writer.write(data)
    shutil.copy(meta_file, output_meta_file)

def process_tar_chunk(chunk, input_root, output_root, save_dir_keypoints, save_dir_dwpose_mp4):
    for wds_path in chunk:
        rel_path = os.path.relpath(wds_path, input_root)
        output_wds_path = os.path.join(output_root, rel_path)
        reshape_dwpose_to_wds(wds_path, output_wds_path, save_dir_keypoints, save_dir_dwpose_mp4)
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0908',
                        help='Input root')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0908_reshape',
                        help='Output root')
    parser.add_argument('--max_processes', type=int, default=32,
                        help='Max processes')

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

    processes = []  # 存储进程的列表
    max_processes = args.max_processes

    input_dir = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    output_dir = os.path.join(args.output_root, os.path.basename(os.path.normpath(video_root)))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Split wds_list into chunks
    input_tar_paths = glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True)
    chunk_size = len(input_tar_paths) // max_processes
    if len(input_tar_paths) % max_processes != 0:
        chunk_size += 1
    chunks = [input_tar_paths[i:i + chunk_size] for i in range(0, len(input_tar_paths), chunk_size)]

    # 测试
    # process_tar_chunk(chunks[0], input_dir, output_dir, save_dir_keypoints, save_dir_dwpose_reshape_mp4)
    
    for chunk_idx, chunk in enumerate(chunks):
        p = Process(
            target=process_tar_chunk,
            args=(chunk, input_dir, output_dir, save_dir_keypoints, save_dir_dwpose_reshape_mp4)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        gc.collect()






    
