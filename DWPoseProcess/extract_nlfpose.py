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
import decord
from NLFPoseExtract.process_nlf import process_video_nlf, preview_nlf_as_images
import pickle

# 总体逻辑：从wds中读视频，然后存一个3d kpts 字典到 pt文件
# 同时要存2D可视化之后的NLF -> 全部存放

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
        key = r['__key__']
        mp4_bytes = r.get("mp4", None)
        motion_indices = r.get("motion_indices", None)
            
                
        try:
            decord.bridge.set_bridge("torch")
            vr = VideoReader(io.BytesIO(mp4_bytes))   # 这里都是原视频，没有动的
            frames = vr.get_batch(range(len(vr)))
            _, video_height, video_width, _ = frames.shape
            frames = torch.from_numpy(frames) if type(frames) is not torch.Tensor else frames
        except Exception as e:
            print(e)
            print('load video error: ', key)
            continue
        item = {'__key__': key, 'frames': frames, 'height': video_height, 'width': video_width, 'motion_indices': motion_indices}
        yield item


def producer_worker_wds(tar_paths, task_queue):
    for tar_path in tar_paths:
        produce_nlfpose(tar_path, task_queue)
    

def produce_nlfpose(wds_path, task_queue):
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
        task_queue.put(data)

def gpu_worker(task_queue, save_dir_smpl, save_dir_smpl_preview):
    model = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()
    while True:
        item = task_queue.get()
        if item is None:
            break
        try:
            frames = item['frames']
            key = item['__key__']
            height = item['height']
            width = item['width']
            motion_indices = item['motion_indices']
            output_data = process_video_nlf(model, frames, height, width)
            output_data['motion_indices'] = motion_indices
            with open(os.path.join(save_dir_smpl, key + '.pkl'), 'wb') as f:
                pickle.dump(output_data, f)
            # vis_images = preview_nlf_as_images(output_data)
            # save_videos_from_pil(vis_images, os.path.join(save_dir_smpl_preview, key + '.mp4'))
        except Exception as e:
            print(f"Task failed: {e}")


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
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank')
    parser.add_argument('--world_size', type=int, default=1,
                        help='World size')

    args = parser.parse_args()
    config = load_config(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)

    video_root = config.get('video_root', '')
    

    save_dir_smpl = os.path.join(video_root, 'smpl')
    save_dir_smpl_preview = os.path.join(video_root, 'smpl_preview')
    os.makedirs(save_dir_smpl, exist_ok=True)
    # os.makedirs(save_dir_smpl_preview, exist_ok=True)


    processes = []  # 存储进程的列表
    max_queue_size = 1
    task_queue = multiprocessing.Queue(maxsize=max_queue_size)
    # Split wds_list into chunks
    input_root = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    input_tar_paths = glob.glob(os.path.join(input_root, "**", "*.tar"), recursive=True)
    input_tar_paths = sorted(input_tar_paths)
    input_tar_paths_for_the_rank = input_tar_paths[args.local_rank::args.world_size]
    p = multiprocessing.Process(target=gpu_worker, args=(task_queue, save_dir_smpl, save_dir_smpl_preview))
    p.start()

    producer_worker_wds(input_tar_paths_for_the_rank, task_queue)
    for _ in range(max_queue_size):
        task_queue.put(None)
    
    p.join(timeout=6000)
    if p.is_alive():
        print("Warning: GPU worker process did not finish within the expected time")
        p.terminate()
    

    






    
