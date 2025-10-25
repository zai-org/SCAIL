import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
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


def process_video_nlf(model, vr_frames, bboxes):
    # Ensure output directory exists
    # pose_results = {
    #     'joints3d_nonparam': [],
    # }
    pose_meta_list = []
    vr_frames = vr_frames.cuda()
    height, width = vr_frames.shape[1], vr_frames.shape[2]
    result_list = []

    batch_size = 64
    buffer = torch.zeros(
        (batch_size, height, width, 3),
        dtype=vr_frames.dtype,
        device='cuda'
    )
    buffer_count = 0
    with torch.inference_mode(), torch.device('cuda'):
        for frame, bbox_list in zip(vr_frames, bboxes):
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox
                x1_px = max(0, math.floor(x1 * width - width * 0.025))
                y1_px = max(0, math.floor(y1 * height - height * 0.05))
                x2_px = min(width, math.ceil(x2 * width + width * 0.025))
                y2_px = min(height, math.ceil(y2 * height + height * 0.05))

                cropped_region = frame[y1_px:y2_px, x1_px:x2_px, :]
                buffer[buffer_count, y1_px:y2_px, x1_px:x2_px, :] = cropped_region
                buffer_count += 1

                # 一旦 buffer 满了，推理并清空
                if buffer_count == batch_size:
                    frame_batch = buffer.permute(0, 3, 1, 2)
                    pred = model.detect_smpl_batched(frame_batch)
                    if 'joints3d_nonparam' in pred:
                        result_list.extend(pred['joints3d_nonparam'])
                    else:
                        result_list.extend([None] * buffer_count)

                    buffer.zero_()
                    buffer_count = 0

        # 处理最后不满一批的残余
        if buffer_count > 0:
            frame_batch = buffer[:buffer_count].permute(0, 3, 1, 2)
            pred = model.detect_smpl_batched(frame_batch)
            if 'joints3d_nonparam' in pred:
                result_list.extend(pred['joints3d_nonparam'])
            else:
                result_list.extend([None] * buffer_count)

    index = 0
    for bbox_list in bboxes:
        n = len(bbox_list)
        pose_meta_list.append({"video_height": height, "video_width": width, "bboxes": bbox_list, "nlfpose": result_list[index : index + n]})
        index += n
    
    del buffer               # 删除 Python 引用
    torch.cuda.empty_cache()
    return pose_meta_list


def process_video_nlf_original(model, vr_frames):
    # Ensure output directory exists
    # pose_results = {
    #     'joints3d_nonparam': [],
    # }
    pose_meta_list = []
    vr_frames = vr_frames.cuda()
    height, width = vr_frames.shape[1], vr_frames.shape[2]
    result_list = []
    people_count_list = []

    batch_size = 64
    buffer = torch.zeros(
        (batch_size, height, width, 3),
        dtype=vr_frames.dtype,
        device='cuda'
    )
    buffer_count = 0
    with torch.inference_mode(), torch.device('cuda'):
        for frame in vr_frames:
            buffer[buffer_count] = frame
            buffer_count += 1

            # 一旦 buffer 满了，推理并清空
            if buffer_count == batch_size:
                frame_batch = buffer.permute(0, 3, 1, 2)
                pred = model.detect_smpl_batched(frame_batch)
                if 'joints3d_nonparam' in pred:
                    result_list.extend(pred['joints3d_nonparam'])
                else:
                    result_list.extend([None] * buffer_count)

                buffer.zero_()
                buffer_count = 0

        # 处理最后不满一批的残余
        if buffer_count > 0:
            frame_batch = buffer[:buffer_count].permute(0, 3, 1, 2)
            pred = model.detect_smpl_batched(frame_batch)
            if 'joints3d_nonparam' in pred:
                result_list.extend(pred['joints3d_nonparam'])
            else:
                result_list.extend([None] * buffer_count)

    index = 0
    for index in range(len(vr_frames)):
        pose_meta_list.append({"video_height": height, "video_width": width, "bboxes": None, "nlfpose": result_list[index]})
    
    del buffer               # 删除 Python 引用
    torch.cuda.empty_cache()
    return pose_meta_list


def process_fn_video(src, bbox_dir):
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
            frames = torch.from_numpy(frames) if type(frames) is not torch.Tensor else frames
            bbox_path = os.path.join(bbox_dir, key + '.pt')
            if os.path.exists(bbox_path):
                bboxes = torch.load(bbox_path)
            else:
                print('no bboxes file: ', key)
                continue
        except Exception as e:
            print(e)
            print('load video error: ', key)
            continue
        item = {'__key__': key, 'frames': frames, 'bboxes': bboxes}
        yield item


def producer_worker_wds(tar_paths, save_dir_bbox, task_queue):
    for tar_path in tar_paths:
        produce_nlfpose(tar_path, save_dir_bbox, task_queue)
    

def produce_nlfpose(wds_path, save_dir_bbox, task_queue):
    dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_video, bbox_dir=save_dir_bbox),
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
            bboxes = item['bboxes']
            output_data = process_video_nlf(model, frames, bboxes)
            
            with open(os.path.join(save_dir_smpl, key + '.pkl'), 'wb') as f:
                pickle.dump(output_data, f)
        except Exception as e:
            print(f"Task failed: {e}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# def process_tar_debug(wds_path):
#     model = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()
#     dataset = wds.DataPipeline(
#             wds.SimpleShardList(wds_path, seed=None),
#             wds.tarfile_to_samples(),
#             partial(process_fn_video),
#         )
#     dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: x[0])
#     for data in tqdm(dataloader):
#         item = data
#         if item is None:
#             break
#         try:
#             frames = item['frames']
#             key = item['__key__']
#             bboxes = torch.load(os.path.join(save_dir_bbox, key + '.pt'))
#             output_data = process_video_nlf(model, frames, bboxes)
            
#             with open(os.path.join(save_dir_smpl, key + '.pkl'), 'wb') as f:
#                 pickle.dump(output_data, f)
            
#         except Exception as e:
#             print(f"Task failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_pack_wds_0923add_step1',
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
    save_dir_bbox = os.path.join(video_root, 'bboxes')
    os.makedirs(save_dir_smpl, exist_ok=True)


    processes = []  # 存储进程的列表
    max_queue_size = 32
    task_queue = multiprocessing.Queue(maxsize=max_queue_size)
    # Split wds_list into chunks
    input_root = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    input_tar_paths = glob.glob(os.path.join(input_root, "**", "*.tar"), recursive=True)
    input_tar_paths = sorted(input_tar_paths)
    input_tar_paths_for_the_rank = input_tar_paths[args.local_rank::args.world_size]

    # 并行流程
    p = multiprocessing.Process(target=gpu_worker, args=(task_queue, save_dir_smpl))
    p.start()

    producer_worker_wds(input_tar_paths_for_the_rank, save_dir_bbox, task_queue)
    for _ in range(max_queue_size):
        task_queue.put(None)
    
    p.join(timeout=6000)
    if p.is_alive():
        print("Warning: GPU worker process did not finish within the expected time")
        p.terminate()

    # 串行debug
    # for wds_path in input_tar_paths_for_the_rank:
    #     process_tar_debug(wds_path)

    

    






    
