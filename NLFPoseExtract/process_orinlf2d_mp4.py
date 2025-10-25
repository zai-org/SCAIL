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
from NLFPoseExtract.nlf_render import render_nlf_as_images
from DWPoseProcess.dwpose import DWposeDetector
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from DWPoseProcess.extract_nlfpose import process_video_nlf_original
from pose_draw.draw_pose_main import draw_pose_to_canvas_np
from NLFPoseExtract.reshape_utils_3d import reshapePool3d
from NLFPoseExtract.nlf_draw import preview_nlf_2d_ori
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy


if __name__ == '__main__':
    model_nlf = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()
    detector = DWposeDetector(use_batch=False).to(0)


    # evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"
    # evaluation_dir = "/workspace/ys_data/evaluation_hard/eval_data"
    evaluation_dir = "/workspace/ys_data/evaluation_multiple_human_v2/eval_data"
    decord.bridge.set_bridge("torch")

    for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
        if subdir.startswith('.'):
            continue
        # if subdir != "005":
        #     continue
        # mp4_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        mp4_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        # out_path_aligned = os.path.join(evaluation_dir, subdir, 'smpl_hybrid_aligned.mp4')
        out_path_ori = os.path.join(evaluation_dir, subdir, 'hybrid_nlf2d.mp4')
        meta_cache_dir = os.path.join(evaluation_dir, subdir, 'meta')
        poses_cache_path = os.path.join(meta_cache_dir, 'keypoints.pt')
        det_cache_path = os.path.join(meta_cache_dir, 'bboxes.pt')
        nlf_cache_path = os.path.join(meta_cache_dir, 'nlf_results.pkl')
        os.makedirs(meta_cache_dir, exist_ok=True)

        vr = VideoReader(mp4_path)
        vr_frames = vr.get_batch(list(range(len(vr))))
        height, width = vr_frames.shape[1], vr_frames.shape[2]
        ori_frame_list = []
        for vr_frame in vr_frames:
            ori_frame_list.append(vr_frame.cpu().numpy())


        
        nlf_results = process_video_nlf_original(model_nlf, vr_frames)
        detector_return_list = []
        # 逐帧解码
        pil_frames = []
        for i in range(len(vr_frames)):
            pil_frame = Image.fromarray(vr_frames[i].numpy())
            pil_frames.append(pil_frame)
            detector_result = detector(pil_frame)
            detector_return_list.append(detector_result)
        poses, _, _ = zip(*detector_return_list) # 这里存的是整个视频的poses
        
        frames_np_rgba = preview_nlf_2d_ori(nlf_results)

        canvas_2d = draw_pose_to_canvas_np(poses, pool=None, H=height, W=width, reshape_scale=0, show_feet_flag=False, show_body_flag=False, show_cheek_flag=True, dw_hand=True)
        for i in range(len(frames_np_rgba)):
            frame_img = frames_np_rgba[i]
            canvas_img = canvas_2d[i]
            mask = canvas_img != 0
            frame_img[:, :, :3][mask] = canvas_img[mask]
            frames_np_rgba[i] = frame_img

        mpy.ImageSequenceClip(frames_np_rgba, fps=16).write_videofile(out_path_ori)

        

