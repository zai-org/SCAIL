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

def process_tar(wds_chunk, output_root, rank_id):
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
            partial(process_fn_video, meta_dict=meta_dict),
        )
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=lambda x: x[0])
        data_iter = iter(dataloader)

        for data in data_iter:
            try:
                key = data['__key__']
                motion_indices = data['motion_indices']
                data.pop('motion_indices', None)
                if len(motion_indices) > 81:
                    obj = meta_dict.get(key, None)
                    data_1 = copy.deepcopy(data)
                    obj_1 = copy.deepcopy(obj)
                    data_2 = copy.deepcopy(data)
                    obj_2 = copy.deepcopy(obj)
                    data_1['__key__'] = key + '_part1'
                    obj_1['key'] = key + '_part1'
                    data_2['__key__'] = key + '_part2'
                    obj_2['key'] = key + '_part2'
                    obj_1['motion_indices'] = motion_indices[:81]
                    obj_2['motion_indices'] = motion_indices[81:]
                    obj_2['ref_image_indices'] = motion_indices[70:81]
                    obj_list.append(obj_1)
                    sample_list.append(data_1)
                    obj_list.append(obj_2)
                    sample_list.append(data_2)
                else:
                    obj = meta_dict.get(key, None)
                    new_data = copy.deepcopy(data)
                    new_data['__key__'] = key + '_old'
                    obj['key'] = key + "_old"
                    obj_list.append(obj)
                    sample_list.append(new_data)

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
        



if __name__ == "__main__":
    import argparse
    from multiprocessing import Process

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929_step5_1013',
                        help='Input root')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_1013_step5_final',
                        help='Output root')

    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    # 找到所有 .tar 文件
    input_tar_paths = sorted(glob.glob(os.path.join(input_root, "**", "*.tar"), recursive=True))

    num_processes = 32
    chunk_size = math.ceil(len(input_tar_paths) / num_processes)
    chunks = [input_tar_paths[i:i + chunk_size] for i in range(0, len(input_tar_paths), chunk_size)]

    # 启动 32 个进程
    processes = []
    for rank, wds_chunk in enumerate(chunks):
        p = Process(target=process_tar, args=(wds_chunk, output_root, rank))
        p.start()
        processes.append(p)
        print(f"Started process {rank} with {len(wds_chunk)} files")

    # 等待所有进程结束
    for p in processes:
        p.join()

    print("All processes completed.")









    
