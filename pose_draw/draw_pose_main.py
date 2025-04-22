import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pose_draw.draw_utils as util
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil, resize_image
from DWPoseProcess.checkUtils import *
import random
import shutil
import argparse
import yaml  # Add this import
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from draw_3d_utils import *
from reshape_utils import *


def draw_pose_points_only(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]   # subset是认为的有效点

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose_points_only(canvas, candidate, subset)

    canvas = util.draw_handpose_points_only(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)
    return canvas

def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]   # subset是认为的有效点

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if len(subset[0]) <= 18:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    else:
        canvas = util.draw_bodypose_with_feet(canvas, candidate, subset)

    canvas = util.draw_handpose_lr(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)
    return canvas

def draw_pose_to_canvas(poses, pool, H, W, reshape_flag, points_only_flag):
    canvas_lst = []
    for pose in poses:
        if reshape_flag:
            pool.apply_random_reshapes(pose)
        if points_only_flag:
            canvas = draw_pose_points_only(pose, H, W)
        else:
            canvas = draw_pose(pose, H, W)
        canvas_img = Image.fromarray(canvas)
        canvas_lst.append(canvas_img)
    return canvas_lst


def get_filenames_from_directory(dwpose_keypoints_dir, threed_keypoints_dir):
    mp4_filenames_dwpose = []
    mp4_filenames_3dpose = []
    # 通过keypoints和mp4的交集取所有可用的mp4
    if dwpose_keypoints_dir:
        for root, dirs, files in os.walk(dwpose_keypoints_dir):
            for file in files:
                if file.lower().endswith('.pt'):  # 只查找 .mp4 文件
                    mp4_filenames_dwpose.append(file.replace(".pt", ".mp4"))  # 获取绝对路径
    if threed_keypoints_dir:
        for root, dirs, files in os.walk(threed_keypoints_dir):
            for file in files:
                if file.lower().endswith('.jsonl'):
                    mp4_filenames_3dpose.append(file.replace(".jsonl", ".mp4"))
    
    return mp4_filenames_dwpose, mp4_filenames_3dpose


def get_poses_from_keypoints(mp4_path, dwpose_keypoint_path, threed_keypoint_path, pose_type):
    if "dwpose+3dpose" in pose_type:
        poses_dwpose = torch.load(dwpose_keypoint_path)
        poses_3dpose = read_pose_from_jsonl(threed_keypoint_path)
        poses = [poses_dwpose, poses_3dpose]
        # poses = mix_3d_poses(poses_dwpose, poses_3dpose)

    elif "dwpose" in pose_type:
        poses = torch.load(dwpose_keypoint_path)
    elif "3dpose" in pose_type:
        poses = read_pose_from_jsonl(threed_keypoint_path)

    return poses



def process_video(mp4_path, dwpose_keypoint_path, threed_keypoint_path, reshape_flag, points_only_flag, wanted_fps=None, output_dirname=None, pose_type="dwpose"):
    frames, fps = read_frames_and_fps_as_np(mp4_path)
    initial_frame = frames[0]
    poses = get_poses_from_keypoints(mp4_path, dwpose_keypoint_path, threed_keypoint_path, pose_type)
    output_path = os.path.join(output_dirname, os.path.basename(mp4_path))
    pool = reshapePool(alpha=0.6)
    if len(poses) == 2:
        canvas_lst_0 = draw_pose_to_canvas(poses[0], pool, initial_frame.shape[0], initial_frame.shape[1], reshape_flag, points_only_flag)
        canvas_lst_1 = draw_pose_to_canvas(poses[1], pool, initial_frame.shape[0], initial_frame.shape[1], reshape_flag, points_only_flag)
        canvas_lst = []
        for canvas_0, canvas_1 in zip(canvas_lst_0, canvas_lst_1):
            # 把两个PIL Image相加
            canvas = Image.blend(canvas_0, canvas_1, 0.38)
            canvas_lst.append(canvas)

    else:
        canvas_lst = draw_pose_to_canvas(poses, pool, initial_frame.shape[0], initial_frame.shape[1], reshape_flag, points_only_flag)
    os.makedirs(output_dirname, exist_ok=True)
    save_videos_from_pil(canvas_lst, output_path, wanted_fps)


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
    reshape_flag = config.get("reshape_flag", False)
    points_only_flag = config.get("points_only_flag", False)
    remove_last_flag = config.get("remove_last_flag", False)
    pose_type = config.get("pose_type", "dwpose")

    if reshape_flag and points_only_flag:
        raise Exception("reshape_flag and points_only_flag cannot be both True")
    elif reshape_flag:
        representation_dirname = f"_{pose_type}_reshaped_filtered"
    elif points_only_flag:
        representation_dirname = f"_{pose_type}_points_filtered"
    else:
        representation_dirname = f"_{pose_type}_filtered"


    mp4_paths = []
    dwpose_keypoint_paths = []
    threed_keypoint_paths = []

    for directory in directories:
        filtered_representation_dir = directory + representation_dirname # TODO: 这里要改
        if remove_last_flag:
            # 删除 directory 中所有文件
            if os.path.exists(filtered_representation_dir):
                shutil.rmtree(filtered_representation_dir)
            print(f"已清除上次产生的{filtered_representation_dir}文件夹")

        video_directory_name = directory.split("/")[-1]

        # video_directory_name 是 directory的最后一层子目录
        dwpose_keypoints_dir, threed_keypoints_dir = None, None
        if "dwpose" in pose_type:
            dwpose_keypoints_dir = directory.replace(video_directory_name, "videos_keypoints")
        if "3dpose" in pose_type:
            threed_keypoints_dir = directory.replace(video_directory_name, "videos_keypoints_3dpose")
        mp4_filenames_dwpose, mp4_filenames_3dpose = get_filenames_from_directory(dwpose_keypoints_dir, threed_keypoints_dir)

        print(f"Processing directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if "dwpose+3dpose" in pose_type:
                    if file in mp4_filenames_dwpose and file in mp4_filenames_3dpose and file.lower().endswith('.mp4'):
                        full_path = os.path.join(root, file)  # 获取绝对路径
                        full_dwpose_path = os.path.join(dwpose_keypoints_dir, file.replace(".mp4", ".pt"))
                        full_threed_path = os.path.join(threed_keypoints_dir, file.replace(".mp4", ".jsonl"))
                        mp4_paths.append(full_path)
                        dwpose_keypoint_paths.append(full_dwpose_path)
                        threed_keypoint_paths.append(full_threed_path)

                elif "dwpose" in pose_type:
                    if file in mp4_filenames_dwpose and file.lower().endswith('.mp4'):  # 只查找 .mp4 文件
                        full_path = os.path.join(root, file)  # 获取绝对路径
                        full_dwpose_path = os.path.join(dwpose_keypoints_dir, file.replace(".mp4", ".pt"))
                        mp4_paths.append(full_path)
                        dwpose_keypoint_paths.append(full_dwpose_path)
                        threed_keypoint_paths.append(None)
                elif "3dpose" in pose_type:
                    if file in mp4_filenames_3dpose and file.lower().endswith('.mp4'):  # 只查找 .mp4 文件
                        full_path = os.path.join(root, file)  # 获取绝对路径
                        full_threed_path = os.path.join(threed_keypoints_dir, file.replace(".mp4", ".jsonl"))
                        mp4_paths.append(full_path)
                        dwpose_keypoint_paths.append(None)
                        threed_keypoint_paths.append(full_threed_path)
    # 串行
        for path_idx, mp4_path in tqdm(enumerate(mp4_paths), desc="Processing videos", unit="video"):
            process_video(mp4_path, dwpose_keypoint_paths[path_idx], threed_keypoint_paths[path_idx], reshape_flag, points_only_flag, wanted_fps=16, output_dirname=filtered_representation_dir, pose_type=pose_type)
    # 并行
    # with Pool(64) as p:
    #     p.starmap(process_video, [(mp4_path, reshape_flag, points_only_flag, original_resolution_flag, 16, True, representation_dirname) for mp4_path in mp4_paths])


    


