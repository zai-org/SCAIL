import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.AAUtils import save_videos_from_pil
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
from extract_dwpose import convert_scores_to_specific_bboxes

def draw_bbox_to_mp4(frames_PIL, bboxes):
    # 输入: frames_PIL: T of PIL Images, bboxes: T of list of (x1, y1, x2, y2), x, y 属于 [0, 1]
    # 输出: 在frames_PIL上用红色框标出bboxes，并返回out_PIL，为T的PIL Image 
    from PIL import ImageDraw
    
    out_PIL = []
    
    for frame_idx, (frame, frame_bboxes) in enumerate(zip(frames_PIL, bboxes)):
        # 复制原图像以避免修改原始数据
        frame_copy = frame.copy()
        draw = ImageDraw.Draw(frame_copy)
        
        W, H = frame.size
        
        for bbox in frame_bboxes:
            if bbox is None or len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # 将归一化坐标转换为像素坐标
            x1_pixel = int(x1 * W)
            y1_pixel = int(y1 * H)
            x2_pixel = int(x2 * W)
            y2_pixel = int(y2 * H)
            
            # 绘制红色矩形框，线宽为3
            draw.rectangle([x1_pixel, y1_pixel, x2_pixel, y2_pixel], 
                          outline='red', width=3)
        
        out_PIL.append(frame_copy)
    
    return out_PIL

def process_single_video(detector, frames_tensor, out_path_mp4):

    detector_return_list = []

    # 逐帧解码
    pil_frames = []
    for i in range(len(frames_tensor)):
        pil_frame = Image.fromarray(frames_tensor[i].numpy())
        pil_frames.append(pil_frame)
        detector_result = detector(pil_frame)
        detector_return_list.append(detector_result)


    W, H = pil_frames[0].size

    poses, scores, det_results = zip(*detector_return_list) # 这里存的是整个视频的poses
    mp4_results = draw_pose_to_canvas(poses, pool=None, H=H, W=W, reshape_scale=0, points_only_flag=False, show_feet_flag=False, dw_hand=True, show_body_flag=False, show_cheek_flag=True, show_face_flag=False)

    save_videos_from_pil(mp4_results, out_path_mp4, fps=16)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
            
        
if __name__ == "__main__":
    local_rank = 7
    detector = DWposeDetector(use_batch=False).to(local_rank)
    evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"
    decord.bridge.set_bridge("torch")


    for subdir_idx, subdir in tqdm(enumerate(os.listdir(evaluation_dir))):
        ori_video_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        out_path_mp4 = os.path.join(evaluation_dir, subdir, 'cheek_hands.mp4')
        vr = VideoReader(ori_video_path)
        frames = vr.get_batch(list(range(len(vr))))
        frames = torch.from_numpy(frames) if type(frames) is not torch.Tensor else frames
        process_single_video(detector, frames, out_path_mp4)

    

    
