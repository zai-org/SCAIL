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

def pack_render_to_wds(wds_path, output_wds_path, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render, save_dir_smpl_render_aug, keys_all):
    obj_list = []
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
    output_meta_file = output_wds_path.replace('.tar', '.meta.jsonl')
    dataset = wds.DataPipeline(
            wds.SimpleShardList(wds_path, seed=None),
            wds.tarfile_to_samples(),
            partial(process_fn_video, meta_dict=meta_dict),
        )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False, collate_fn=lambda x: x[0])
    if os.path.exists(output_wds_path):   # 提前清除旧tar
        os.remove(output_wds_path)
    with TarWriter(output_wds_path) as writer:
        for data in tqdm(dataloader):
            key = data['__key__']
            if key not in keys_all:
                continue
            else:
                print(f"process {key}")
            # motion_indices = data['motion_indices']
            try:
                smpl_rendered_path = os.path.join(save_dir_smpl_render, key + '.mp4')
                smpl_rendered_aug_path = os.path.join(save_dir_smpl_render_aug, key + '.mp4')
                smpl_rendered_noface_path = os.path.join(save_dir_smpl_render_aug.replace('smpl_render_aug', 'smpl_render_noface'), key + '.mp4')
                if not os.path.exists(smpl_rendered_path) or not os.path.exists(smpl_rendered_aug_path) or not os.path.exists(smpl_rendered_noface_path):
                    print(f"skip {key}, no smpl rendered")
                    continue
                
                with open(smpl_rendered_path, "rb") as f:
                    smpl_render_data = f.read()

                with open(smpl_rendered_aug_path, "rb") as f:
                    smpl_render_data_aug = f.read()
                
                with open(smpl_rendered_noface_path, "rb") as f:
                    smpl_render_data_noface = f.read()

                data['append_smpl_render'] = smpl_render_data
                data['append_smpl_render_aug'] = smpl_render_data_aug
                data['append_smpl_render_noface'] = smpl_render_data_noface

                data.pop('motion_indices')
                obj_list.append(meta_dict.get(key, None))
                writer.write(data)
            except Exception as e:
                print(e)
                print(f"skip {key}, error")
                continue
    with open(output_meta_file, 'w', encoding='utf-8') as outfile:
        writer = jsonlines.Writer(outfile)
        writer.write_all(obj_list)
        writer.close()
        

def process_tar_chunk(chunk, input_root, output_root, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render, save_dir_smpl_render_aug, keys_all):
    for wds_path in chunk:
        rel_path = os.path.relpath(wds_path, input_root)
        output_wds_path = os.path.join(output_root, rel_path)
        pack_render_to_wds(wds_path, output_wds_path, save_dir_keypoints, save_dir_dwpose_mp4, save_dir_smpl, save_dir_smpl_render, save_dir_smpl_render_aug, keys_all)
        gc.collect()
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='DWPoseExtractConfig/gymnastics_mb_v2025091601.yaml', 
                        help='Path to YAML configuration file')
    parser.add_argument('--input_root', type=str, default='/workspace/ywh_data/pose_packed_wds_0929add_step3',
                        help='Input root')
    parser.add_argument('--output_root', type=str, default='/workspace/ywh_data/pose_packed_wds_1024_step5',
                        help='Output root')

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
    save_dir_smpl_render_aug = os.path.join(video_root, 'smpl_render_aug')


    keys_mp4 = collect_keys_from_dirs()
    keys_jsonl = collect_keys_from_jsonl()
    keys_all = keys_mp4 + keys_jsonl

    input_dir = os.path.join(args.input_root, os.path.basename(os.path.normpath(video_root)))
    output_dir = os.path.join(args.output_root, os.path.basename(os.path.normpath(video_root)))
    os.makedirs(output_dir, exist_ok=True)
    # Split wds_list into chunks
    input_tar_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True))
    process_tar_chunk(input_tar_paths, input_dir, output_dir, save_dir_keypoints, save_dir_dwpose_reshape_mp4, save_dir_smpl, save_dir_smpl_render, save_dir_smpl_render_aug, keys_all)







    
