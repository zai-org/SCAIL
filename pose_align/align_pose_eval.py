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
from pose_align.eval_align_utils import run_align_video

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

    canvas = util.draw_handpose(canvas, hands)  # 测试要用旧版手

    canvas = util.draw_facepose(canvas, faces)
    return canvas

def draw_pose_to_canvas(poses, H, W):
    canvas_lst = []
    for pose in poses:
        canvas = draw_pose(pose, H, W)
        canvas_img = Image.fromarray(canvas)
        canvas_lst.append(canvas_img)
    return canvas_lst


def draw_keypoints_with_align(image_keypoints_path, video_keypoints_path, image_path, video_path, output_path):
    frames, fps = read_frames_and_fps_as_np(video_path)
    # 打开 image_path 得到 initial_frame
    ref_frame = cv2.imread(image_path)
    initial_frame = frames[0]
    poses_image = torch.load(image_keypoints_path)[0]
    poses_video = torch.load(video_keypoints_path)
    # poses = run_align_video(initial_frame.shape[0], initial_frame.shape[1], ref_frame.shape[0], ref_frame.shape[1], poses_image, poses_video)
    poses = poses_video
    canvas_lst = draw_pose_to_canvas(poses, ref_frame.shape[0], ref_frame.shape[1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_videos_from_pil(canvas_lst, output_path, fps=fps)


if __name__ == "__main__":
    evaluation_dir = "/workspace/ywh_data/cross_pair_eval100"
    for test_case_dir in tqdm(sorted(os.listdir(evaluation_dir))):
        if not os.path.isdir(os.path.join(evaluation_dir, test_case_dir)):
            continue
        if not test_case_dir.endswith("keypoints"):
            continue
        image_keypoints_path, video_keypoints_path = None, None
        for image_video in os.listdir(os.path.join(evaluation_dir, test_case_dir)):
            if image_video.endswith("_ref.pt"):
                image_keypoints_path = os.path.join(evaluation_dir, test_case_dir, image_video)
            elif image_video.endswith(".pt"):
                video_keypoints_path = os.path.join(evaluation_dir, test_case_dir, image_video)
        if os.path.exists(image_keypoints_path) and os.path.exists(video_keypoints_path):
            video_sub_dir = test_case_dir.replace("_keypoints", "")
            video_path = os.path.join(evaluation_dir, video_sub_dir, os.path.basename(video_keypoints_path).replace(".pt", ".mp4"))
            image_path = os.path.join(evaluation_dir, video_sub_dir, os.path.basename(image_keypoints_path).replace("_ref.pt", ".jpg"))
            output_path = os.path.join(evaluation_dir, video_sub_dir, os.path.basename(video_keypoints_path).replace(".pt", "_unaligned.mp4"))
            # original_path = os.path.join(evaluation_dir, video_sub_dir, "dwpose_keypoints_with_align.mp4")
            # if os.path.exists(original_path):
            #     # original_path 是个文件，不是文件夹 
            #     os.remove(original_path)
            draw_keypoints_with_align(image_keypoints_path, video_keypoints_path, image_path, video_path, output_path)


    


