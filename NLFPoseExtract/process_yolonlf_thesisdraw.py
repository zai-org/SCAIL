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
from DWPoseProcess.extract_nlfpose import process_video_nlf
from NLFPoseExtract.reshape_utils_3d import reshapePool3d
import copy
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy



if __name__ == '__main__':
    model_nlf = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()

    evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"
    # evaluation_dir = "/workspace/yanwenhao/dwpose_draw/multi_test/results"
    decord.bridge.set_bridge("torch")

    for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
        if subdir != "010":
            continue
        mp4_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        out_path = os.path.join(evaluation_dir, subdir, 'smpl_hybrid_test.mp4')
        meta_cache_dir = os.path.join(evaluation_dir, subdir, 'meta')
        poses_cache_path = os.path.join(meta_cache_dir, 'keypoints.pt')
        det_cache_path = os.path.join(meta_cache_dir, 'bboxes.pt')
        nlf_cache_path = os.path.join(meta_cache_dir, 'nlf_results.pkl')
        os.makedirs(meta_cache_dir, exist_ok=True)

        vr = VideoReader(mp4_path)
        vr_frames = vr.get_batch(list(range(len(vr))))
        height, width = vr_frames.shape[1], vr_frames.shape[2]


        if os.path.exists(poses_cache_path) and os.path.exists(det_cache_path) and os.path.exists(nlf_cache_path):
            poses = torch.load(poses_cache_path)
            bboxes = torch.load(det_cache_path)
            with open(nlf_cache_path, 'rb') as f:
                nlf_results = pickle.load(f)

            reshapepool = reshapePool3d(reshape_type="high", height=height, width=width)
            # body_pool = ["elf", "random_long_arm_long_leg", "king-kong", "test_case"]
            body_pool = ["test_case"]
            for body_type in body_pool:
                reshapepool.aug_init(body_type)
                frames_ori_np = render_nlf_as_images(copy.deepcopy(nlf_results), poses, reshape_pool=reshapepool, aug_2d=False)
                mpy.ImageSequenceClip(frames_ori_np, fps=16).write_videofile(out_path.replace('.mp4', f'_{body_type}.mp4'))
        else:
            detector = DWposeDetector(use_batch=False).to(0)
            detector_return_list = []

            # 逐帧解码
            pil_frames = []
            for i in range(len(vr_frames)):
                pil_frame = Image.fromarray(vr_frames[i].numpy())
                pil_frames.append(pil_frame)
                detector_result = detector(pil_frame)
                detector_return_list.append(detector_result)


            W, H = pil_frames[0].size

            poses, scores, det_results = zip(*detector_return_list) # 这里存的是整个视频的poses
            nlf_results = process_video_nlf(model_nlf, vr_frames, det_results)

            torch.save(poses, poses_cache_path)
            torch.save(det_results, det_cache_path)
            with open(nlf_cache_path, 'wb') as f:
                pickle.dump(nlf_results, f)

        

