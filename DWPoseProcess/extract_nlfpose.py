import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from DWPoseProcess.AAUtils import save_videos_from_pil
from DWPoseProcess.checkUtils import *
from collections import deque
import shutil
import torch
import yaml
from pose_draw.draw_pose_main import draw_pose_to_canvas, reshapePool
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
import decord


def process_video_nlf(model, vr_frames, video_height, video_width):
    # Ensure output directory exists
    pose_results = {
        'joints3d_nonparam': [],
    }

    with torch.inference_mode(), torch.device('cuda'):
        batch_size = 64
        for i in range(0, len(vr_frames), batch_size):
            frame_batch = vr_frames[i:i+batch_size].cuda().permute(0,3,1,2)
            pred = model.detect_smpl_batched(frame_batch)
            if 'joints3d_nonparam' in pred:
                #pose_results[key].append(pred[key].cpu().numpy())
                pose_results['joints3d_nonparam'].extend(pred['joints3d_nonparam'])
            else:
                pose_results['joints3d_nonparam'].extend([None] * len(pred['joints3d_nonparam']))

    # Prepare output data
    output_data = {
        'video_length': len(vr_frames),
        'video_width': video_width,
        'video_height': video_height,
        'pose': pose_results
    }
    return output_data


def process_fn_video(src):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue
        key = r['__key__']
        mp4_bytes = r.get("mp4", None)

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
        item = {'__key__': key, 'frames': frames, 'height': video_height, 'width': video_width}
        yield item


def producer_worker_wds(tar_paths, task_queue):
    for tar_path in tar_paths:
        produce_nlfpose(tar_path, task_queue)
    

def produce_nlfpose(wds_path, task_queue):
    dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_video),
        )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: x[0])
    for data in tqdm(dataloader):
        task_queue.put(data)

def gpu_worker(task_queue, save_dir_smpl):
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
            output_data = process_video_nlf(model, frames, height, width)
            with open(os.path.join(save_dir_smpl, key + '.pkl'), 'wb') as f:
                pickle.dump(output_data, f)
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
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_pack_wds_0923_step1',
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
    os.makedirs(save_dir_smpl, exist_ok=True)


    processes = []  # 存储进程的列表
    max_queue_size = 32
    task_queue = multiprocessing.Queue(maxsize=max_queue_size)
    # Split wds_list into chunks
    input_root = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    input_tar_paths = glob.glob(os.path.join(input_root, "**", "*.tar"), recursive=True)
    input_tar_paths = sorted(input_tar_paths)
    input_tar_paths_for_the_rank = input_tar_paths[args.local_rank::args.world_size]
    p = multiprocessing.Process(target=gpu_worker, args=(task_queue, save_dir_smpl))
    p.start()

    producer_worker_wds(input_tar_paths_for_the_rank, task_queue)
    for _ in range(max_queue_size):
        task_queue.put(None)
    
    p.join(timeout=6000)
    if p.is_alive():
        print("Warning: GPU worker process did not finish within the expected time")
        p.terminate()
    

    






    
