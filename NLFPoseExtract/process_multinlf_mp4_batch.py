import os
import sys
import cv2
import torch
import pickle
import torchvision
import shutil
import glob
import random
from tqdm import tqdm
import decord
from decord import VideoReader, cpu, gpu
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
from NLFPoseExtract.nlf_render import render_multi_nlf_as_images
from DWPoseProcess.dwpose import DWposeDetector
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from DWPoseProcess.extract_nlfpose import process_video_multi_nlf
from NLFPoseExtract.reshape_utils_3d import reshapePool3d
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy



def get_largest_bbox_indices(bboxes, num_bboxes=2):
    # 计算每个bbox的面积
    def calculate_area(bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    # 计算每个bbox的面积，并保留原索引
    bboxes_with_area = [(i, calculate_area(bbox)) for i, bbox in enumerate(bboxes)]
    
    # 根据面积从大到小排序
    bboxes_with_area.sort(key=lambda x: x[1], reverse=True)
    
    # 取出面积最大的 num_bboxes 个索引
    largest_indices = [idx for idx, _ in bboxes_with_area[:num_bboxes]]
    
    return largest_indices


def change_poses_to_limit_num(poses, bboxes):
    for pose, bbox in zip(poses, bboxes):
        human_num = len(bbox)
        largest_indices = get_largest_bbox_indices(bbox, num_bboxes=3)
        for i in range(human_num):
            if i not in largest_indices:
                pose['bodies']['subset'][i:i+1][:] = -1
                pose['hands'][2*i:2*i+2][:] = -1
                pose['faces'][i:i+1][:] = -1




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank')
    parser.add_argument('--world_size', type=int, default=1,
                        help='World size')

    args = parser.parse_args()


    model_nlf = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()
    decord.bridge.set_bridge("torch")

    samurai_root = "/workspace/ywh_data/DataProcessNew/complexMovements_bilibili_v2025102800/samurai_videos"
    meta_root = "/workspace/ywh_data/DataProcessNew/complexMovements_bilibili_v2025102800/raw_videos_meta"
    output_root = "/workspace/ywh_data/DataProcessNew/complexMovements_bilibili_v2025102800/smpl_render"
    os.makedirs(output_root, exist_ok=True)
    keys = sorted(os.listdir(samurai_root))
    input_keys = keys[args.local_rank::args.world_size]

    for key in input_keys:
        try:
            # if subdir != "005":
            #     continue
            out_path = os.path.join(output_root, f"{key}.mp4")
            meta_cache_dir = os.path.join(meta_root, key)
            poses_cache_path = os.path.join(meta_cache_dir, 'keypoints.pt')
            det_cache_path = os.path.join(meta_cache_dir, 'bboxes.pt')

            vr_frames_list = []
            for samurai_mp4_path in sorted(glob.glob(os.path.join(samurai_root, key, '*.mp4'))):
                vr_tmp = VideoReader(samurai_mp4_path)
                if len(vr_tmp) != 65 and len(vr_tmp) != 81:
                    print(f"Video {samurai_mp4_path} has {len(vr_tmp)} frames, expected 65 or 81, skipping.")
                    continue
                vr_frames_tmp = vr_tmp.get_batch(list(range(len(vr_tmp))))
                vr_frames_list.append(vr_frames_tmp)
            if len(vr_frames_list) == 0:
                print(f"No video files found for {key}, skipping.")
                continue
            if os.path.exists(meta_cache_dir):
                poses = torch.load(poses_cache_path)
                bboxes = torch.load(det_cache_path)
                change_poses_to_limit_num(poses, bboxes)
                nlf_results = process_video_multi_nlf(model_nlf, vr_frames_list)
                frames_ori_np = render_multi_nlf_as_images(nlf_results, poses, reshape_pool=None)
                mpy.ImageSequenceClip(frames_ori_np, fps=16).write_videofile(out_path)
        except Exception as e:
            print(f"Error processing {key}: {e}")
            traceback.print_exc()

        

