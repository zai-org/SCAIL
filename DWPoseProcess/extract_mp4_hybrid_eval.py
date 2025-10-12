import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from collections import deque
import shutil
import torch
import yaml
from pose_draw.draw_pose_main import draw_pose_to_canvas_np, draw_pose, scale_image_hw_keep_size
from extractUtils import check_single_human_requirements, check_multi_human_requirements, human_select
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, TimeoutError, as_completed
from AAUtils import read_frames
from decord import VideoReader
from fractions import Fraction
import io
import gc
from PIL import Image
from multiprocessing import Process
import decord
import json
import glob
import sys
from VITPoseExtract.pipeline import VITPosePipeline, get_cond_images
import cv2
import copy
from pose_draw.reshape_utils import reshapePool
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy


def process_single_video(detector_dwpose, detector_vitpose, frames_np, out_path_mp4, out_path_mp4_cheek):

    detector_return_list = []

    # 逐帧解码
    pil_frames = []
    for i in range(len(frames_np)):
        pil_frame = Image.fromarray(frames_np[i])
        pil_frames.append(pil_frame)
        detector_result = detector_dwpose(pil_frame)
        detector_return_list.append(detector_result)


    W, H = pil_frames[0].size

    poses, scores, det_results = zip(*detector_return_list) # 这里存的是整个视频的poses
    tpl_pose_metas = detector_vitpose(frames_np)

    np_results = get_hybrid_video(frames_np[0], poses, tpl_pose_metas, H, W, reshape_scale=0, only_cheek=False, draw_hand=False, dw_bgr=True)
    # np_results_cheek = get_hybrid_video(frames_np[0], poses, tpl_pose_metas, H, W, reshape_scale=0, only_cheek=True)
    # pool = reshapePool(alpha=0.6)
    # np_results = draw_pose_to_canvas_np(copy.deepcopy(poses), pool=None, H=H, W=W, reshape_scale=0)
    # np_results_cheek = draw_pose_to_canvas_np(copy.deepcopy(poses), pool=None, H=H, W=W, reshape_scale=0, show_body_flag=False, show_cheek_flag=True)
    print("save video to ", out_path_mp4)
    mpy.ImageSequenceClip(np_results, fps=16).write_videofile(out_path_mp4.replace('hybrid.mp4', 'hybrid_animate.mp4'))
    # breakpoint()
    # print("save video to ", out_path_mp4_cheek)
    # mpy.ImageSequenceClip(np_results_cheek, fps=16).write_videofile(out_path_mp4_cheek)


def get_hybrid_video(first_frame_np, poses, tpl_pose_metas, H, W, reshape_scale=0.6, only_cheek=True, draw_hand=False, dw_bgr=False):
    if reshape_scale > 0:
        pool = reshapePool(alpha=reshape_scale)
        for pose, tpl_pose_meta in zip(poses, tpl_pose_metas):
            pool.apply_random_reshapes(pose, tpl_pose_meta)

    cond_images = get_cond_images(first_frame_np, tpl_pose_metas, only_cheek=only_cheek, draw_hand=draw_hand, dw_bgr=dw_bgr)
    for cond_image, pose in zip(cond_images, poses):
        show_face = True
        if reshape_scale > 0:
            if random.random() < 0.08:
                show_face = False
        canvas = draw_pose(pose, H, W, show_feet=False, show_body=False, show_hand=True, show_face=show_face, show_cheek=False, optimized_face=True) # H W 3 np
        mask = (canvas.sum(axis=2) > 0)   # shape: (H, W), bool
        if reshape_scale > 0:
            if random.random() < 0.08:
                cond_image[:] = 0
        cond_image[mask] = canvas[mask]

    if reshape_scale > 0:
        if random.random() < 0.3:
        # if True:
        #     print("do scale")
            scale_h = random.uniform(0.85, 1.04)
            scale_w = random.uniform(0.85, 1.04)
            for cond_image in cond_images:
                new_cond_image = scale_image_hw_keep_size(cond_image, scale_h, scale_w)
                cond_image[:] = new_cond_image


    return cond_images

    

    

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
            
        
if __name__ == "__main__":
    local_rank = 7
    detector_dwpose = DWposeDetector(use_batch=False).to(local_rank)
    detector_vitpose = VITPosePipeline(det_checkpoint_path="/workspace/yanwenhao/Wan2.2/Wan2.2-Animate-14B/process_checkpoint/det/yolov10m.onnx", pose2d_checkpoint_path="/workspace/yanwenhao/Wan2.2/Wan2.2-Animate-14B/process_checkpoint/pose2d/vitpose_h_wholebody.onnx")
    evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"


    for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
        ori_video_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        out_path_mp4 = os.path.join(evaluation_dir, subdir, 'hybrid.mp4')
        out_path_mp4_cheek = os.path.join(evaluation_dir, subdir, 'hybrid_cheek.mp4')
        vr = VideoReader(ori_video_path)
        frames_np = vr.get_batch(list(range(len(vr)))).asnumpy()
        process_single_video(detector_dwpose, detector_vitpose, frames_np, out_path_mp4, out_path_mp4_cheek)

    

    
