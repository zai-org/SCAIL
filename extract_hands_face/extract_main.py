import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pose_draw.draw_utils as util
from pose_draw.draw_3d_utils import *
from pose_draw.reshape_utils import *
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil
from DWPoseProcess.checkUtils import *
import random
import shutil
import argparse
import yaml  # Add this import
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from decord import VideoReader
import copy

def points_to_bbox(points, relative_min_threshold=0.05, around_padding=0.005):
    """
    根据给定的点集计算最小外接矩形（Bounding Box）。

    参数:
        points (np.ndarray): 一个 [N, 2] 形状的 NumPy 数组，其中 N 是点的数量。
                            每个点的坐标都应在 [0, 1] 范围内。

    返回:
        tuple: 包含 Bounding Box 坐标的元组，格式为 (x_min, y_min, x_max, y_max)。
               如果输入点集为空，则返回 (0, 0, 0, 0)。
    """
    if not isinstance(points, np.ndarray) or points.shape[0] == 0:
        return (0, 0, 0, 0)

    # 找到所有点的最小 x 和 y 坐标
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])

    # 找到所有点的最大 x 和 y 坐标
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    if x_min < around_padding or y_min < around_padding or x_max > 1 - around_padding or y_max > 1 - around_padding:
        return (0, 0, 0, 0)
    
    if y_max - y_min < relative_min_threshold or x_max - x_min < relative_min_threshold:
        return (0, 0, 0, 0)

    # 返回 bbox 的坐标
    return (x_min - around_padding, y_min - around_padding, x_max + around_padding, y_max + around_padding)
    

def crop_and_save_frame(frame, bbox_relative, save_path='cropped_image.png', pixel_min_threshold=16):
    """
    从图像帧中裁剪指定区域并保存，使用相对边界框坐标。

    参数:
        frame (np.ndarray): 一个形状为 [height, width, channels] 的 NumPy 数组，表示图像帧。
        bbox_relative (tuple): 一个包含相对边界框坐标的元组 (x1_rel, y1_rel, x2_rel, y2_rel)。
                               这些值应在 0.0 到 1.0 之间。
        save_path (str, 可选): 保存裁剪后图像的路径。默认为 'cropped_image.png'。
    """
    # 获取图像帧的尺寸
    height, width, _ = frame.shape

    # 从相对坐标元组中解包
    x1_rel, y1_rel, x2_rel, y2_rel = bbox_relative

    # 将相对坐标转换为绝对像素坐标
    # 使用 int() 进行类型转换，确保坐标是整数
    x1 = int(x1_rel * width)
    y1 = int(y1_rel * height)
    x2 = int(x2_rel * width)
    y2 = int(y2_rel * height)

    if x2 - x1 < pixel_min_threshold or y2 - y1 < pixel_min_threshold:
        return
    
    # 将 NumPy 数组转换为 PIL Image 对象
    img = Image.fromarray(frame)

    # 裁剪图像
    cropped_img = img.crop((x1, y1, x2, y2))

    # 保存裁剪后的图像
    cropped_img.save(save_path)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video directories based on YAML config')
    parser.add_argument('--config', type=str, default='video_directories.yaml', 
                        help='Path to YAML configuration file')
    args = parser.parse_args()
    # Load configuration
    config = load_config(args.config)
    directories = config.get("directories")
    for directory in directories:
        print(f"Processing directory: {directory}")
        parent_dir = os.path.dirname(directory)
        sub_dir = os.path.basename(directory)
        keypoint_directory = os.path.join(parent_dir, "videos_multi_step2_filtered_keypoints") if "multi" in sub_dir else os.path.join(parent_dir, "videos_step2_filtered_keypoints")    # TODO: 修改完
        bbox_directory = os.path.join(parent_dir, "videos_multi_step2_filtered_bboxes") if "multi" in sub_dir else os.path.join(parent_dir, "videos_step2_filtered_bboxes")    # TODO: 修改完
        out_hand_directory = os.path.join(parent_dir.replace("DataProcessNew", "BlurTest"), "extracted_hands")
        out_face_directory = os.path.join(parent_dir.replace("DataProcessNew", "BlurTest"), "extracted_faces")
        out_lip_directory = os.path.join(parent_dir.replace("DataProcessNew", "BlurTest"), "extracted_lips")
        out_human_directory = os.path.join(parent_dir.replace("DataProcessNew", "BlurTest"), "extracted_humans")
        out_video_directory = os.path.join(parent_dir.replace("DataProcessNew", "BlurTest"), "extracted_videos")
        if os.path.exists(out_hand_directory) and os.path.exists(out_face_directory):
            shutil.rmtree(out_hand_directory)
            shutil.rmtree(out_face_directory)
            shutil.rmtree(out_human_directory)
            shutil.rmtree(out_lip_directory)
            shutil.rmtree(out_video_directory)
        os.makedirs(out_hand_directory, exist_ok=True)
        os.makedirs(out_face_directory, exist_ok=True)
        os.makedirs(out_human_directory, exist_ok=True)
        os.makedirs(out_lip_directory, exist_ok=True)
        os.makedirs(out_video_directory, exist_ok=True)

        files = os.listdir(directory)
        random_files = random.sample(files, 250)

        for mp4_idx, mp4_file in tqdm(enumerate(random_files)):
            mp4_path = os.path.join(directory, mp4_file)
            keypoint_path = os.path.join(keypoint_directory, mp4_file.replace('.mp4', '.pt'))
            bbox_path = os.path.join(bbox_directory, mp4_file.replace('.mp4', '.pt'))
            frames_np, _ = read_frames_and_fps_as_np(mp4_path)  # frames_np: h. w. c
            keypoints_frames = torch.load(keypoint_path)
            bboxes_frames = torch.load(bbox_path)


            shutil.copy(mp4_path, os.path.join(out_video_directory, f"{mp4_file}"))
            for idx, keypoints_per_frame in enumerate(keypoints_frames):
                frame = frames_np[idx]
                hands_kpts = keypoints_per_frame['hands']
                face_kpts = keypoints_per_frame['faces']
                human_bbox = bboxes_frames[idx]
                # for hands_idx, hands_kpt in enumerate(hands_kpts):
                #     bbox = points_to_bbox(hands_kpt)
                #     print(f"crop {mp4_file} frame {idx}")
                #     crop_and_save_frame(frame, bbox, save_path=os.path.join(out_hand_directory, f"{mp4_file.replace('.mp4', '')}_{idx}_{hands_idx}.png"))
                # for faces_idx, face_kpt in enumerate(face_kpts):
                #     bbox = points_to_bbox(face_kpt)
                #     print(f"crop {mp4_file} frame {idx}")
                #     crop_and_save_frame(frame, bbox, save_path=os.path.join(out_face_directory, f"{mp4_file.replace('.mp4', '')}_{idx}_{faces_idx}.png"))
                # for faces_idx, face_kpt in enumerate(face_kpts):
                #     bbox = points_to_bbox(face_kpt[48:68], relative_min_threshold=0.001)
                #     print(f"crop {mp4_file} frame {idx}")
                #     crop_and_save_frame(frame, bbox, save_path=os.path.join(out_lip_directory, f"{mp4_file.replace('.mp4', '')}_{idx}_{faces_idx}.png"), pixel_min_threshold=4)
                # for bbox_idx, bbox in enumerate(human_bbox):
                #     crop_and_save_frame(frame, bbox, save_path=os.path.join(out_human_directory, f"{mp4_file.replace('.mp4', '')}_{idx}_{bbox_idx}.png"))
                if idx % 4 == 1:
                    img = Image.fromarray(frame)
                    frame_sep_dir = os.path.join(out_video_directory, f"{mp4_file}".replace('.mp4', ""))
                    # os.makedirs(frame_sep_dir, exist_ok=True)
                    # 保存裁剪后的图像
                    # img.save(os.path.join(frame_sep_dir, f"{idx}.png"))




        


