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
from pose_align.eval_align_utils import run_align_video
from pose_draw.draw_pose_main import draw_pose_to_canvas


def draw_keypoints_with_align(image_keypoints_path, video_keypoints_path, image_path, video_path, output_path):
    frames, fps = read_frames_and_fps_as_np(video_path)
    # 打开 image_path 得到 initial_frame
    ref_frame = cv2.imread(image_path)
    initial_frame = frames[0]
    poses_image = torch.load(image_keypoints_path)[0]
    poses_video = torch.load(video_keypoints_path)
    poses = run_align_video(initial_frame.shape[0], initial_frame.shape[1], ref_frame.shape[0], ref_frame.shape[1], poses_image, poses_video)
    canvas_lst = draw_pose_to_canvas(poses, pool=None, H=ref_frame.shape[0], W=ref_frame.shape[1], reshape_scale=0, points_only_flag=False, show_feet_flag=False, show_hand_flag=False, show_face_flag=True, dw_bgr=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_videos_from_pil(canvas_lst, output_path, fps=fps)


if __name__ == "__main__":
    evaluation_dir = "/workspace/ywh_data/EvalCross/cross_pair_eval100"
    for subdir in sorted(os.listdir(evaluation_dir)):
        driving_mp4_path = os.path.join(evaluation_dir, subdir, "GT.mp4")
        ref_jpg = os.path.join(evaluation_dir, subdir, "ref_image.jpg")
    num_str_list = [""]
    for num_str in num_str_list:
        print(f"processing {num_str}")
        driving_name = os.listdir(os.path.join(evaluation_dir, num_str, "videos"))[0].replace(".mp4", "")
        ref_name = os.listdir(os.path.join(evaluation_dir, num_str, "ref_image"))[0].replace(".jpg", "").replace(".png", "")
        image_keypoints_path = os.path.join(evaluation_dir, num_str, "ref_image_video_keypoints", f"{ref_name}.pt")
        video_keypoints_path = os.path.join(evaluation_dir, num_str, "videos_keypoints", f"{driving_name}.pt")
        image_path = os.path.join(evaluation_dir, num_str, "ref_image", f"{ref_name}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(evaluation_dir, num_str, "ref_image", f"{ref_name}.png")
        video_path = os.path.join(evaluation_dir, num_str, "videos", f"{driving_name}.mp4")
        output_dir = os.path.join(evaluation_dir, num_str, "videos_aligned")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{driving_name}.mp4")
        draw_keypoints_with_align(image_keypoints_path, video_keypoints_path, image_path, video_path, output_path)


    


