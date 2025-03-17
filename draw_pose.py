import cv2
import numpy as np
from PIL import Image
import utils as util
import torch
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil
import random
import shutil
import argparse
import yaml  # Add this import
import os


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

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)
    return canvas


def pose_reshape(alpha, pose, 
                 anchor_point, anchor_part, end_point, end_part, 
                 affected_body_points, affected_faces_points, affected_left_hands_points, affected_right_hands_points):
    """
    直接修改原始pose中的某些点的位置，通过根据一个不动点和端点之间的向量偏移被影响点的位置。
    一次只能有一个anchor_point, 一个end_point，多个affected_points
    alpha: 变化比例
    anchor_point: 起始点
    anchor_part: 起始点所属的身体部分，0代表body candidate, 1代表face, 2代表hand
    end_point: 中止点
    end_part: 中止点所属的身体部分，0代表body candidate, 1代表face, 2代表hand
    """
    bodies = pose["bodies"]
    faces = pose["faces"][0]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"][0]   # subset是认为的有效点
    # 取0，概率最大的那个

    assert anchor_part == 0, "anchor part must belong to the body"
    anchor_x, anchor_y = candidate[anchor_point]
    if subset[anchor_point] == -1:
        return

    if end_part == 0:
        end_x, end_y = candidate[end_point]
        if subset[end_point] == -1:
            return
    elif end_part == 1:
        end_x, end_y = faces[end_point]
    # 不考虑Hands
    if end_x == -1 and end_y == -1:
        return
    
    vector_x = end_x - anchor_x
    vector_y = end_y - anchor_y
    offset_x = vector_x * alpha
    offset_y = vector_y * alpha
    
    for affected_body_point in affected_body_points:
        if subset[affected_body_point] == -1:
            continue
        affected_x, affected_y = candidate[affected_body_point]
        candidate[affected_body_point] = (affected_x + offset_x, affected_y + offset_y)
    for affected_faces_point in affected_faces_points:
        affected_x, affected_y = faces[affected_faces_point]
        if affected_x == -1 and affected_y == -1:
            continue
        faces[affected_faces_point] = (affected_x + offset_x, affected_y + offset_y)
    for affected_hands_point in affected_left_hands_points:
        affected_x, affected_y = hands[1][affected_hands_point]
        if affected_x == -1 and affected_y == -1:
            continue
        hands[1][affected_hands_point] = (affected_x + offset_x, affected_y + offset_y)
    for affected_hands_point in affected_right_hands_points:
        affected_x, affected_y = hands[0][affected_hands_point]
        if affected_x == -1 and affected_y == -1:
            continue
        hands[0][affected_hands_point] = (affected_x + offset_x, affected_y + offset_y)

class reshapePool:
    def __init__(self, alpha):
        self.faces_indices = np.arange(0, 68)
        self.left_hands_indices = np.arange(0, 21)
        self.right_hands_indices = np.arange(0, 21)
        self.alpha = alpha

        self.reshape_methods = [
            self.reshape_body,
            self.reshape_arm,
            self.reshape_leg,
            self.reshape_shoulder,
            self.reshape_face
        ]
        self.selected_methods = random.sample(self.reshape_methods, 2)
        # self.selected_methods = [self.reshape_face]

    def apply_random_reshapes(self, pose):
        # Apply the two selected reshape methods
        for method in self.selected_methods:
            method(pose)

    def reshape_body(self, pose):
        pose_reshape(self.alpha, pose,
                     1, 0, 8, 0,
                     [8, 9, 10], [], [], []
                     )
        pose_reshape(self.alpha, pose,
                    1, 0, 11, 0,
                    [11, 12, 13], [], [], []
                    )
        
    def reshape_arm(self, pose):
        pose_reshape(self.alpha, pose,
                     2, 0, 3, 0,
                     [3, 4], [], self.left_hands_indices, []
                     )
        pose_reshape(self.alpha, pose,
                    5, 0, 6, 0,
                    [6, 7], [], [], self.right_hands_indices
                    )
        pose_reshape(self.alpha, pose,
                     3, 0, 4, 0,
                     [4], [], self.left_hands_indices, []
                     )
        pose_reshape(self.alpha, pose,
                    6, 0, 7, 0,
                    [7], [], [], self.right_hands_indices
                    )

    def reshape_leg(self, pose):
        pose_reshape(self.alpha, pose,
                     8, 0, 9, 0,
                     [9, 10], [], [], []
                     )
        pose_reshape(self.alpha, pose,
                    11, 0, 12, 0,
                    [12, 13], [], [], []
                    )
        pose_reshape(self.alpha, pose,
                     9, 0, 10, 0,
                     [10], [], [], []
                     )
        pose_reshape(self.alpha, pose,
                    12, 0, 13, 0,
                    [13], [], [], []
                    )
        

    def reshape_shoulder(self, pose):
        pose_reshape(self.alpha, pose,
                     1, 0, 2, 0,
                     [2, 3, 4], [], self.left_hands_indices, []
                     )
        pose_reshape(self.alpha, pose,
                    1, 0, 5, 0,
                    [5, 6, 7], [], [], self.right_hands_indices
                    )
    
    def reshape_face(self, pose):
        for i in self.faces_indices:
            pose_reshape(self.alpha, pose,
                     1, 0, i, 1,
                     [], [i], [], []
                     )
    

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

    directory = config.get("directory")
    reshape_flag = config.get("reshape_flag", False)
    points_only_flag = config.get("points_only_flag", False)

    ori_dir = directory.replace("videos_dwpose","videos_dwpose_ori")
    os.makedirs(ori_dir, exist_ok=True)
    dwpose_paths = []

    for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.mp4'):  # 只查找 .mp4 文件
                    full_path = os.path.join(root, file)  # 获取绝对路径
                    dwpose_paths .append(full_path)

    for dwpose_path in dwpose_paths:
        poses = torch.load(dwpose_path.replace(".mp4", ".pt"))
        pool = reshapePool(alpha=0.6)
        frames, fps = read_frames_and_fps_as_np(dwpose_path)
        H, W, C = frames[0].shape
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
        if reshape_flag and points_only_flag:
            raise Exception("reshape_flag and points_only_flag cannot be both True")
        elif reshape_flag:
            save_videos_from_pil(canvas_lst, dwpose_path.replace("videos_dwpose", "videos_dwpose_reshape"), fps)
        elif points_only_flag:
            save_videos_from_pil(canvas_lst, dwpose_path.replace("videos_dwpose", "videos_points"), fps)
        else:
            save_videos_from_pil(canvas_lst, dwpose_path.replace("videos_dwpose", "videos_dwpose_ori"), fps)


    


