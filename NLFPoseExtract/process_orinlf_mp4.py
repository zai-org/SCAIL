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
from NLFPoseExtract.reshape_utils_3d import reshapePool3d
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy



if __name__ == '__main__':
    model_nlf = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()

    # evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"
    evaluation_dir = "/workspace/yanwenhao/dwpose_draw/multi_test/results"
    decord.bridge.set_bridge("torch")

    for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
        # if subdir != "005":
        #     continue
        # mp4_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        mp4_path = os.path.join(evaluation_dir, subdir, 'raw_video.mp4')
        # out_path_aligned = os.path.join(evaluation_dir, subdir, 'smpl_hybrid_aligned.mp4')
        out_path_ori = os.path.join(evaluation_dir, subdir, 'smpl_hybrid.mp4')
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
        results = render_nlf_as_images(nlf_results, None, reshape_pool=None)
        results = [np.concatenate([results[i][:,:,:3], ori_frame_list[i]], axis=1) for i in range(len(results))]
        mpy.ImageSequenceClip(results, fps=16).write_videofile(out_path_ori)


        

