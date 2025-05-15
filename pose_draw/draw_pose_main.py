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
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil, resize_image
from DWPoseProcess.checkUtils import *
import random
import shutil
import argparse
import yaml  # Add this import
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count



def draw_pose_points_only(pose, H, W, show_feet=False):
    raise NotImplementedError("draw_pose_points_only is not implemented")

def draw_pose(pose, H, W, show_feet=False):
    final_canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for i in range(len(pose["bodies"]["candidate"])):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        bodies = pose["bodies"]
        faces = pose["faces"][i:i+1]
        hands = pose["hands"][2*i:2*i+2]
        candidate = bodies["candidate"][i]
        subset = bodies["subset"][i:i+1]   # subset是认为的有效点

        if len(subset[0]) <= 18 or show_feet == False:
            canvas = util.draw_bodypose(canvas, candidate, subset)
        else:
            canvas = util.draw_bodypose_with_feet(canvas, candidate, subset)

        canvas = util.draw_handpose_lr(canvas, hands)

        canvas = util.draw_facepose(canvas, faces)
        final_canvas = final_canvas + canvas
    return final_canvas

def draw_pose_to_canvas(poses, pool, H, W, reshape_scale, points_only_flag, show_feet_flag):
    canvas_lst = []
    for pose in poses:
        if reshape_scale > 0:
            pool.apply_random_reshapes(pose)
        if points_only_flag:
            canvas = draw_pose_points_only(pose, H, W, show_feet_flag)
        else:
            canvas = draw_pose(pose, H, W, show_feet_flag)
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
        poses = correct_full_pose_from_3d(poses_dwpose, poses_3dpose)

    elif "dwpose" in pose_type:
        poses = torch.load(dwpose_keypoint_path)
    elif "3dpose" in pose_type:
        poses = read_pose_from_jsonl(threed_keypoint_path)

    return poses



def process_video(mp4_path, dwpose_keypoint_path, threed_keypoint_path, reshape_scale, points_only_flag, show_feet_flag, wanted_fps=None, output_dirname=None, pose_type="dwpose"):
    frames, fps = read_frames_and_fps_as_np(mp4_path)
    initial_frame = frames[0]
    poses = get_poses_from_keypoints(mp4_path, dwpose_keypoint_path, threed_keypoint_path, pose_type)
    output_path = os.path.join(output_dirname, os.path.basename(mp4_path))
    pool = reshapePool(alpha=reshape_scale)
    canvas_lst = draw_pose_to_canvas(poses, pool, initial_frame.shape[0], initial_frame.shape[1], reshape_scale, points_only_flag, show_feet_flag)
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
    reshape_scale = config.get("reshape_scale", 0)
    points_only_flag = config.get("points_only_flag", False)
    remove_last_flag = config.get("remove_last_flag", False)
    show_feet_flag = config.get("show_feet_flag", False)
    pose_type = config.get("pose_type", "dwpose")
    target_representation_dirname = config.get("target_representation_suffix", None)
    keypoints_suffix_dwpose = config.get("keypoints_suffix_dwpose", "_keypoints_new")
    keypoints_suffix_3dpose = config.get("keypoints_suffix_3dpose", "_keypoints_3dpose")


    mp4_paths = []
    dwpose_keypoint_paths = []
    threed_keypoint_paths = []

    for directory in directories:
        output_representation_dir = directory + target_representation_dirname
        if remove_last_flag:
            # 删除 directory 中所有文件
            if os.path.exists(output_representation_dir):
                shutil.rmtree(output_representation_dir)
            print(f"已清除上次产生的{output_representation_dir}文件夹")

        video_directory_name = directory.split("/")[-1]

        # video_directory_name 是 directory的最后一层子目录
        dwpose_keypoints_dir, threed_keypoints_dir = None, None
        if "dwpose" in pose_type:
            dwpose_keypoints_dir = directory.replace(video_directory_name, f"{video_directory_name}{keypoints_suffix_dwpose}")   # TODO: 暂时修改
        if "3dpose" in pose_type:
            threed_keypoints_dir = directory.replace(video_directory_name, f"{video_directory_name}{keypoints_suffix_3dpose}")
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
            process_video(mp4_path, dwpose_keypoint_paths[path_idx], threed_keypoint_paths[path_idx], reshape_scale, points_only_flag, show_feet_flag, wanted_fps=16, output_dirname=output_representation_dir, pose_type=pose_type)
    # 并行
        # with Pool(64) as p:
        #     p.starmap(process_video, [(mp4_path, dwpose_keypoint_paths[path_idx], threed_keypoint_paths[path_idx], reshape_scale, points_only_flag, show_feet_flag, 16, output_representation_dir, pose_type) for path_idx, mp4_path in enumerate(mp4_paths)])


    


