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
from pose_draw.draw_pose_main import draw_pose_to_canvas
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

def process_fn_video(src):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue

        yield r.copy()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_tar(wds_chunk, chunk_id, output_root, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, eval_list):
    obj_list = []
    sample_list = []
    shard_size = 100
    shard_id = 0
    output_pattern = "%06d"
    for _, wds_path in tqdm(enumerate(wds_chunk), total=len(wds_chunk), disable=(chunk_id != 0)):
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
            partial(process_fn_video),
        )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
        data_iter = iter(dataloader)

        for data_batch in data_iter:
            data = {}
            for k, v in data_batch.items():
                data[k] = v[0]
            try:
                key = data['__key__']
                multi_path = os.path.join(save_dir_caption_multi, key + '.txt')
                single_path = os.path.join(save_dir_caption, key + '.txt')
                if key in eval_list:
                    print(f"exclude {key}, in eval list")
                    continue

                out_path_bbox = os.path.join(save_dir_bboxes, key + '.pt')
                out_path_hands = os.path.join(save_dir_hands, key + '.pt')
                out_path_faces = os.path.join(save_dir_faces, key + '.pt')
                out_path_mp4 = os.path.join(save_dir_dwpose_mp4, key + '.mp4')

                if os.path.exists(multi_path):
                    with open(multi_path, "r", encoding="utf-8") as f:
                        txt_data = f.read()
                elif os.path.exists(single_path):
                    with open(single_path, "r", encoding="utf-8") as f:
                        txt_data = f.read()
                else:
                    continue
                if os.path.exists(out_path_bbox) and os.path.exists(out_path_hands) and os.path.exists(out_path_faces):
                    pass
                else:
                    continue
                with open(out_path_bbox, "rb") as f:
                    bbox_data = f.read()
                with open(out_path_hands, "rb") as f:
                    hands_data = f.read()
                with open(out_path_faces, "rb") as f:
                    faces_data = f.read()

                obj = meta_dict.get(key, None)
                if obj is None:
                    print(f"skip {key}, no meta")
                    continue
                with open(out_path_mp4, "rb") as f:
                    mp4_data = f.read()
                data['dwpose'] = mp4_data
                data['recaption'] = txt_data
                data['bbox'] = bbox_data
                data['hands'] = hands_data
                data['faces'] = faces_data

                sample_list.append(data)
                obj_list.append(obj)
                
                # Write shard when it reaches the target size
                if len(sample_list) >= shard_size:
                    shard_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.tar'
                    jsonl_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.meta.jsonl'
                    
                    with TarWriter(shard_file) as writer:
                        for sample in sample_list:
                            writer.write(sample)
                    with open(jsonl_file, 'w', encoding='utf-8') as outfile:
                        writer = jsonlines.Writer(outfile)
                        writer.write_all(obj_list)
                        writer.close()
                    
                    print(f"Written shard {shard_id} with {len(sample_list)} samples")
                    sample_list = []
                    obj_list = []
                    shard_id += 1
            
            except Exception as e:
                print(f"Error processing video {key}: {e}")
                continue

    # Write remaining samples if any
    if len(sample_list) > 0:
        shard_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.tar'
        jsonl_file = os.path.join(output_root, output_pattern % chunk_id + '_' + output_pattern % shard_id) + '.meta.jsonl'
        
        with TarWriter(shard_file) as writer:
            for sample in sample_list:
                writer.write(sample)
        with open(jsonl_file, 'w', encoding='utf-8') as outfile:
            writer = jsonlines.Writer(outfile)
            writer.write_all(obj_list)
            writer.close()
        
        print(f"Written final shard {shard_id} with {len(sample_list)} samples")

def get_eval_list():
    eval_list = []
    eval_dirs = ["/workspace/ys_data/evaluation_300/DWPose/videos"]
    clean_eval_dirs = ["/workspace/ywh_data/eval_hq_v2/videos"]
    for eval_dir in eval_dirs:
        for video_name in os.listdir(eval_dir):
            eval_list.append(os.path.splitext(video_name)[0].split('_', 1)[1])
    for clean_eval_dir in clean_eval_dirs:
        for video_name in os.listdir(clean_eval_dir):
            eval_list.append(os.path.splitext(video_name)[0])
    return eval_list

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_default',
                        help='Output root')
    parser.add_argument('--max_processes', type=int, default=8,
                        help='Max processes')

    args = parser.parse_args()
    config = load_config(args.config)

    wds_root = config.get('wds_root', '')
    tar_paths = glob.glob(os.path.join(wds_root, "**", "*.tar"), recursive=True)
    video_root = config.get('video_root', '')
    output_root = os.path.join(args.output_root, os.path.basename(os.path.normpath(video_root)))
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    save_dir_keypoints = os.path.join(video_root, 'keypoints')
    save_dir_bboxes = os.path.join(video_root, 'bboxes')
    save_dir_dwpose_mp4 = os.path.join(video_root, 'dwpose')
    save_dir_hands = os.path.join(video_root, 'hands')
    save_dir_faces = os.path.join(video_root, 'faces')
    save_dir_caption = os.path.join(video_root, 'caption')
    save_dir_caption_multi = os.path.join(video_root, 'caption_multi')

    os.makedirs(save_dir_keypoints, exist_ok=True)
    os.makedirs(save_dir_bboxes, exist_ok=True)
    os.makedirs(save_dir_dwpose_mp4, exist_ok=True)
    os.makedirs(save_dir_hands, exist_ok=True)
    os.makedirs(save_dir_faces, exist_ok=True)
    os.makedirs(save_dir_caption, exist_ok=True)
    os.makedirs(save_dir_caption_multi, exist_ok=True)


    processes = []  # 存储进程的列表
    max_processes = args.max_processes

    # Split wds_list into chunks
    random.shuffle(tar_paths)
    chunk_size = len(tar_paths) // max_processes
    if len(tar_paths) % max_processes != 0:
        chunk_size += 1
    chunks = [tar_paths[i:i + chunk_size] for i in range(0, len(tar_paths), chunk_size)]
    eval_list = get_eval_list()
    
    for chunk_idx, chunk in enumerate(chunks):
        p = Process(
            target=process_tar,
            args=(chunk, chunk_idx, output_root, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_dwpose_mp4, save_dir_caption, save_dir_caption_multi, eval_list)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        gc.collect()






    
