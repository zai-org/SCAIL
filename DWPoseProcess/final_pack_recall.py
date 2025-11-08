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


def collect_keys_from_dirs():
    mp4_ft_dir = "/workspace/ywh_data/ftData_add/finetune_selected"
    mp4_stunt_dir = "/workspace/ywh_data/ftData_add/pose_finetune_preview_0929add/stunt_mb_v2025091601"
    mp4_excluded_dir = "/workspace/ywh_data/ftData_add/finetune_excluded"
    keys_mp4_ft = [key.split(".")[0] for key in os.listdir(mp4_ft_dir)]
    keys_mp4_stunt = [key.split(".")[0] for key in os.listdir(mp4_stunt_dir)]
    keys_mp4_excluded = [key.split(".")[0] for key in os.listdir(mp4_excluded_dir)]
    keys_mp4_ft = list(set(keys_mp4_ft + keys_mp4_stunt) - set(keys_mp4_excluded))
    return keys_mp4_ft

def collect_keys_from_jsonl():
    jsonl_path = "/workspace/ywh_data/ftData/SkelVid_Manual_11249.jsonl"
    with jsonlines.open(jsonl_path) as reader:
        keys = [(os.path.basename(obj["origin_path"])).split(".")[0] for obj in reader if obj["是否符合"] == "符合"]
    return keys

def process_fn_video(src, meta_dict=None):
    worker_info = torch.utils.data.get_worker_info()
    for i, r in enumerate(src):
        if worker_info is not None:
            if i % worker_info.num_workers != worker_info.id:
                continue

        item = r.copy()
        yield item

def process_tar(wds_chunk, output_root, keys_all):
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
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
        data_iter = iter(dataloader)

        for data_batch in data_iter:
            data = {}
            for k, v in data_batch.items():
                data[k] = v[0]
            try:
                key = data['__key__']
                if key not in keys_all:
                    continue
                else:
                    print(f"process {key}")
                    obj = meta_dict.get(key, None)

                # Modify keys to append '_recall'
                new_key = key + '_recall'
                data['__key__'] = new_key
                if obj is not None:
                    obj['key'] = new_key

                sample_list.append(data)
                obj_list.append(obj)
                
                # Write shard when it reaches the target size
                if len(sample_list) >= shard_size:
                    shard_file = os.path.join(output_root, output_pattern % shard_id) + '.tar'
                    jsonl_file = os.path.join(output_root, output_pattern % shard_id) + '.meta.jsonl'

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_finetune_wds_0929_step5_1013',
                        help='Input root')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_1024_step5/recall',
                        help='Output root')

    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)


    keys_mp4 = collect_keys_from_dirs()
    keys_jsonl = collect_keys_from_jsonl()
    keys_all = keys_mp4 + keys_jsonl
    # Split wds_list into chunks
    input_tar_paths = sorted(glob.glob(os.path.join(input_root, "**", "*.tar"), recursive=True))
    process_tar(input_tar_paths, output_root, keys_all)








    
