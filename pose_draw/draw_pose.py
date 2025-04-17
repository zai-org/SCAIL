import cv2
import numpy as np
from PIL import Image
import draw_utils as util
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DWPoseProcess.AAUtils import read_frames_and_fps_as_np, save_videos_from_pil, resize_image
from DWPoseProcess.checkUtils import *
import random
import shutil
import argparse
import yaml  # Add this import
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


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

def convert_3dpose_to_2dpose_body(body_keypoints):
    """
    将20点的3D坐标映射到18点的2D坐标。
    :param poses: 输入的20点坐标列表，每个点为 [x, y, z]
    :return: 映射得到的18点坐标列表，每个点为 [x, y]
    """
    # 映射关系：索引位置
    mapping = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 8: 8, 9: 9, 10: 10,
        14: 11, 15: 12, 16: 13, 11: 18, 12: 19, 13: 20, 17: 21, 18: 22, 19: 23
    }
    
    # 初始化18点坐标列表，默认值为 [-1, -1]
    result = [[-1, -1] for _ in range(24)]
    
    # 遍历映射关系，将对应的20点坐标映射到18点坐标
    for src_idx, dst_idx in mapping.items():
        if src_idx < len(body_keypoints):  # 确保索引不越界
            result[dst_idx] = [body_keypoints[src_idx][1],body_keypoints[src_idx][0]]   # 提取 x, y 坐标
    
    return result

def convert_3dpose_to_2dpose_hand(left_hand_keypoints, right_hand_keypoints, body_keypoints):
    """
    将20点的3D坐标映射到18点的2D坐标。
    :param poses: 输入的20点坐标列表，每个点为 [x, y, z]
    :return: 映射得到的18点坐标列表，每个点为 [x, y]
    """
    # 映射关系：索引位置
    hand_mapping = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
        10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18,
        18: 19, 19: 20
    }

    body_mapping_left = {3: 0}
    body_mapping_right = {6: 0}
    
    # 初始化18点坐标列表，默认值为 [-1, -1]
    left_result = [[-1, -1] for _ in range(21)]
    right_result = [[-1, -1] for _ in range(21)]
    
    # 遍历映射关系，将对应的20点坐标映射到18点坐标
    for src_idx, dst_idx in hand_mapping.items():
        if src_idx < len(left_hand_keypoints):  # 确保索引不越界
            left_result[dst_idx] = [left_hand_keypoints[src_idx][1], left_hand_keypoints[src_idx][0]]  # 提取 x, y 坐标
            right_result[dst_idx] = [right_hand_keypoints[src_idx][1], right_hand_keypoints[src_idx][0]]
    
    for src_idx, dst_idx in body_mapping_left.items():
        if src_idx < len(body_keypoints):
            left_result[dst_idx] = [body_keypoints[src_idx][1], body_keypoints[src_idx][0]]
    for src_idx, dst_idx in body_mapping_right.items():
        if src_idx < len(body_keypoints):
            right_result[dst_idx] = [body_keypoints[src_idx][1], body_keypoints[src_idx][0]]
    
    return [left_result, right_result]

def convert_3dpose_to_2dpose_face(face_keypoints):
    result = [[pt[1], pt[0]] for pt in face_keypoints]
    return result



def read_jsonl(jsonl_path):
    import jsonlines
    poses = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            condidate = convert_3dpose_to_2dpose_body(obj["body"])
            subset = [[ -1,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
        13., -1, -1, -1, -1, 18, 19, 20, 21, 22, 23]]
            faces = [convert_3dpose_to_2dpose_face(obj["face"])]
            hands = convert_3dpose_to_2dpose_hand(obj["left_hand"],obj["right_hand"], obj["body"])
            poses.append({"bodies":{"candidate": condidate, "subset": subset}, "faces": faces, "hands": hands})
    return poses

def process_video(mp4_path, reshape_flag, points_only_flag, wanted_fps=None, representation_dirname="videos_dwpose_filtered", keypoints_dir=None):
    
    frames, fps = read_frames_and_fps_as_np(mp4_path)
    initial_frame = frames[0]
    if "dwpose" in representation_dirname:
        keypoint_path = mp4_path.replace("videos_filtered", "videos_keypoints_filtered").replace(".mp4", ".pt")
        poses = torch.load(keypoint_path)
    elif "3dpose" in representation_dirname:
        # 取mp4_path的文件名
        mp4_name = os.path.basename(mp4_path)
        keypoint_path = os.path.join(keypoints_dir, mp4_name.replace(".mp4", ".jsonl"))
        poses = read_jsonl(keypoint_path)
    pool = reshapePool(alpha=0.6)
    canvas_lst = draw_pose_to_canvas(poses, pool, initial_frame.shape[0], initial_frame.shape[1], reshape_flag, points_only_flag)
    
    # save dwpose
    if "dwpose" in representation_dirname:
        target_representation_path = mp4_path.replace("videos_filtered", representation_dirname)
    elif "3dpose" in representation_dirname:
        target_representation_path = mp4_path.replace("videos", representation_dirname)
    os.makedirs(os.path.dirname(target_representation_path), exist_ok=True)
    save_videos_from_pil(canvas_lst, target_representation_path, wanted_fps)


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

    if reshape_flag and points_only_flag:
        raise Exception("reshape_flag and points_only_flag cannot be both True")
    elif reshape_flag:
        representation_dirname = "videos_3dpose_reshaped_filtered"
    elif points_only_flag:
        representation_dirname = "videos_3dpose_points_filtered"
    else:
        representation_dirname = "videos_3dpose_filtered_test"  # TODO: 这里要改


    mp4_paths = []
    for directory in directories:
        if remove_last_flag:
            # 删除 directory 中所有文件
            # filtered_representation_dir = directory.replace("videos_filtered", representation_dirname) # TODO: 这里要改
            filtered_representation_dir = directory.replace("videos", representation_dirname) # TODO: 这里要改
            if os.path.exists(filtered_representation_dir):
                shutil.rmtree(filtered_representation_dir)
            print("已清除上次产生的")

        keypoint_files = []
        keypoints_dir = directory.replace("videos", "keypoints")
        for root, dirs, files in os.walk(keypoints_dir):
            for file in files:
                if file.lower().endswith('.pt'):  # 只查找 .mp4 文件
                    keypoint_files.append(file.replace(".pt", ".mp4"))  # 获取绝对路径
                elif file.lower().endswith('.jsonl'):
                    keypoint_files.append(file.replace(".jsonl", ".mp4"))

        print(f"Processing directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file in keypoint_files and file.lower().endswith('.mp4'):  # 只查找 .mp4 文件
                    full_path = os.path.join(root, file)  # 获取绝对路径
                    mp4_paths.append(full_path)
    
    # 串行
        for mp4_path in tqdm(mp4_paths, desc="Processing videos", unit="video"):
            process_video(mp4_path, reshape_flag, points_only_flag, wanted_fps=16, representation_dirname=representation_dirname, keypoints_dir=keypoints_dir)
    # 并行
    # with Pool(64) as p:
    #     p.starmap(process_video, [(mp4_path, reshape_flag, points_only_flag, original_resolution_flag, 16, True, representation_dirname) for mp4_path in mp4_paths])


    


