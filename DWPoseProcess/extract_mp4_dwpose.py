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

def process_single_video(detector, frames_tensor, key, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_regional_preview, save_dir_mp4):
    out_path_keypoint = os.path.join(save_dir_keypoints, key + '.pt')
    out_path_bbox = os.path.join(save_dir_bboxes, key + '.pt')
    out_path_hands = os.path.join(save_dir_hands, key + '.pt')
    out_path_faces = os.path.join(save_dir_faces, key + '.pt')
    out_path_regional_preview = os.path.join(save_dir_regional_preview, key + '.mp4')
    out_path_mp4 = os.path.join(save_dir_mp4, key + '.mp4')

    # output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    # if not output_dir.exists():
    #     output_dir.mkdir(parents=True, exist_ok=True)

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
    hands_bboxes = convert_scores_to_specific_bboxes(poses, scores, type='hands', score_threshold=0.6)
    faces_bboxes = convert_scores_to_specific_bboxes(poses, scores, type='faces', score_threshold=0.6)
    mp4_results = draw_pose_to_canvas(poses, pool=None, H=H, W=W, reshape_scale=0, points_only_flag=False, show_feet_flag=False, dw_hand=True)
    preview_results = draw_bbox_to_mp4(pil_frames, hands_bboxes)
    preview_results = draw_bbox_to_mp4(preview_results, faces_bboxes)

    save_videos_from_pil(mp4_results, out_path_mp4, fps=16)
    save_videos_from_pil(preview_results, out_path_regional_preview, fps=16)
    
    torch.save(poses, out_path_keypoint)
    torch.save(det_results, out_path_bbox)
    torch.save(hands_bboxes, out_path_hands)
    torch.save(faces_bboxes, out_path_faces)





def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def produce_mp4(task_queue, raw_video_dir):
    tmp_dir = "/dev/shm/tmp"
    for _, raw_video_filename in tqdm(enumerate(os.listdir(raw_video_dir))):
        if raw_video_filename.endswith('.mp4'):
            decord.bridge.set_bridge("torch")
            vr = VideoReader(os.path.join(raw_video_dir, raw_video_filename))
            frame_indices = list(range(len(vr)))
            frames = vr.get_batch(frame_indices)
            frames = torch.from_numpy(frames) if type(frames) is not torch.Tensor else frames
            key = raw_video_filename.replace('.mp4', '')
            torch.save(frames, os.path.join(tmp_dir, key + '.pt'))
            task_queue.put(key)
            
        
if __name__ == "__main__":
    video_root = "/workspace/ywh_data/evaluation_80_clear"
    local_rank = 0

    save_dir_keypoints = os.path.join(video_root, 'videos_keypoints')
    save_dir_bboxes = os.path.join(video_root, 'videos_bboxes')
    save_dir_hands = os.path.join(video_root, 'videos_hands')
    save_dir_faces = os.path.join(video_root, 'videos_faces')
    save_dir_regional_preview = os.path.join(video_root, 'videos_regional_preview')
    save_dir_mp4 = os.path.join(video_root, 'videos_dwpose_ori')
    raw_mp4_dir =  os.path.join(video_root, 'videos')
    tmp_dir = os.path.join("/dev/shm/tmp")

    os.makedirs(save_dir_keypoints, exist_ok=True)
    os.makedirs(save_dir_bboxes, exist_ok=True)
    os.makedirs(save_dir_hands, exist_ok=True)
    os.makedirs(save_dir_faces, exist_ok=True)
    os.makedirs(save_dir_regional_preview, exist_ok=True)
    os.makedirs(save_dir_mp4, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    detector = DWposeDetector(use_batch=False).to(local_rank)


    tmp_dir = "/dev/shm/tmp"
    for _, raw_video_filename in tqdm(enumerate(os.listdir(raw_mp4_dir))):
        if raw_video_filename.endswith('.mp4'):
            decord.bridge.set_bridge("torch")
            vr = VideoReader(os.path.join(raw_mp4_dir, raw_video_filename))
            frame_indices = list(range(len(vr)))
            frames = vr.get_batch(frame_indices)
            frames = torch.from_numpy(frames) if type(frames) is not torch.Tensor else frames
            key = raw_video_filename.replace('.mp4', '')
            process_single_video(detector, frames, key, save_dir_keypoints, save_dir_bboxes, save_dir_hands, save_dir_faces, save_dir_regional_preview, save_dir_mp4)


    

    
