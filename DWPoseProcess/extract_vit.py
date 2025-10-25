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
from VITPoseExtract.pipeline import VITPosePipeline
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

def  extract_vit(detector_vitpose, wds_path, save_dir_vitpose):
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
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, collate_fn=lambda x: x[0])
    
    for data in tqdm(dataloader):
        key = data['__key__']
        mp4_bytes = data['mp4']
        motion_indices = data['motion_indices']
        vr = VideoReader(io.BytesIO(mp4_bytes))   # h w c
        frames_np_motion = vr.get_batch(motion_indices).asnumpy()
        try:
            tpl_pose_metas_motion = detector_vitpose(frames_np_motion)
            out_path = os.path.join(save_dir_vitpose, f"{key}.pt")
            torch.save(tpl_pose_metas_motion, out_path)
        except Exception as e:
            print(e)
            print('detector error: ', key)
            continue


def process_tar_chunk(detector_vitpose, chunk, input_root, save_dir_vitpose):
    for wds_path in chunk:
        rel_path = os.path.relpath(wds_path, input_root)
        extract_vit(detector_vitpose, wds_path, save_dir_vitpose)
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
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929_step5',
                        help='Output root')
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
    save_dir_vitpose = os.path.join(video_root, 'vitpose')
    os.makedirs(save_dir_vitpose, exist_ok=True)

    processes = []  # 存储进程的列表
    current_process = args.current_process
    max_processes = args.max_processes

    input_dir = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    input_tar_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True))

    
    current_tar_paths = input_tar_paths[current_process::max_processes]
    if len(current_tar_paths) == 0:
        print("No chunks to process")
    detector_vitpose = VITPosePipeline(det_checkpoint_path="/workspace/yanwenhao/Wan2.2/Wan2.2-Animate-14B/process_checkpoint/det/yolov10m.onnx", pose2d_checkpoint_path="/workspace/yanwenhao/Wan2.2/Wan2.2-Animate-14B/process_checkpoint/pose2d/vitpose_h_wholebody.onnx")
    process_tar_chunk(detector_vitpose, current_tar_paths, input_dir, save_dir_vitpose)







    
