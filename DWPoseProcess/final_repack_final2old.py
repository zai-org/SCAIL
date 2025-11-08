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
        yield item

def get_all_keys(wds_root):
    key_set = set()
    wds_paths = glob.glob(os.path.join(wds_root, '*.tar'))
    for wds_path in tqdm(wds_paths, desc="Collecting keys"):
        dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
        )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: x[0])
        data_iter = iter(dataloader)

        for data in data_iter:
            key_set.add(data['__key__'])
    return key_set

def process_tar(wds_chunk, output_root, rank_id, excluded_keys=None):
    obj_list = []
    sample_list = []
    shard_size = 100
    shard_id = 0
    output_pattern = "%06d"
    for _, wds_path in tqdm(enumerate(wds_chunk), total=len(wds_chunk)):
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
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: x[0])
        data_iter = iter(dataloader)

        for data in data_iter:
            try:
                key = data['__key__']
                obj = meta_dict.get(key, None)
                    
                if key.replace("_old", "").replace("_part1", "").replace("_part2", "") in excluded_keys:
                    print(f"skip excluded key {key}")
                    continue
                
                else:
                    sample_list.append(data)
                    obj_list.append(obj)
                
                # Write shard when it reaches the target size
                if len(sample_list) >= shard_size:
                    shard_file = os.path.join(output_root, output_pattern % rank_id + '_' + output_pattern % shard_id) + '.tar'
                    jsonl_file = os.path.join(output_root, output_pattern % rank_id + '_' + output_pattern % shard_id) + '.meta.jsonl'

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
                traceback.print_exc()
                continue

    # Write the last shard if it exists
    if sample_list:
        shard_file = os.path.join(output_root, output_pattern % rank_id + '_' + output_pattern % shard_id) + '.tar'
        jsonl_file = os.path.join(output_root, output_pattern % rank_id + '_' + output_pattern % shard_id) + '.meta.jsonl'

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

if __name__ == "__main__":
    import argparse
    from multiprocessing import Process

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_1101_step5_final',
                        help='Output root')

    args = parser.parse_args()

    input_root = '/workspace/ywh_data/pose_packed_wds_1013_step5_final'
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    excludes_root = '/workspace/ywh_data/pose_packed_wds_0929_step5_1013/biaobei_CN_JP_KR'
    excluded_keys = get_all_keys(excludes_root)

    # 找到所有 .tar 文件
    input_tar_paths = glob.glob(os.path.join(input_root, '*.tar'))
    # process_tar(input_tar_paths, output_root, rank_id=0, excluded_keys=excluded_keys)

    num_processes = 32
    chunk_size = math.ceil(len(input_tar_paths) / num_processes)
    chunks = [input_tar_paths[i:i + chunk_size] for i in range(0, len(input_tar_paths), chunk_size)]

    processes = []
    for rank, wds_chunk in enumerate(chunks):
        p = Process(target=process_tar, args=(wds_chunk, output_root, rank, excluded_keys))
        p.start()
        processes.append(p)
        print(f"Started process {rank} with {len(wds_chunk)} files")

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("All processes completed.")









    
