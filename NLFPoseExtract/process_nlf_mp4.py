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

def process_video_dwpose(model, frames):

    detector_return_list = []

    # 逐帧解码
    pil_frames = []
    for i in range(len(frames)):
        pil_frame = Image.fromarray(frames[i].numpy())
        pil_frames.append(pil_frame)
        detector_result = model(pil_frame)
        detector_return_list.append(detector_result)
    
    poses, _, _ = zip(*detector_return_list)

    return poses     # a list of poses, each pose is a dict, has bodies, faces, hands



if __name__ == '__main__':
    # model_nlf = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()
    # detector = DWposeDetector(use_batch=False).to(0)

    # evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"
    root_dir = '/workspace/ywh_data/DataProcessNew/bili_dance_hengping_250328'
    decord.bridge.set_bridge("torch")


    # for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
    kpt_dir = os.path.join(root_dir, 'keypoints')
    smpl_dir = os.path.join(root_dir, 'smpl')
    out_dir = '/workspace/yanwenhao/dwpose_draw/render_preview'
    

    for smpl_pkl in tqdm(random.sample(os.listdir(smpl_dir), 100)):
        key = smpl_pkl.split('.')[0]
        smpl_pkl_path = os.path.join(smpl_dir, smpl_pkl)
        with open(smpl_pkl_path, 'rb') as f:
            smpl_data = pickle.load(f)
        dwpose_kpt_seq = os.path.join(kpt_dir, key + '.pt')
        dwpose_kpt_seq = torch.load(dwpose_kpt_seq)
        out_path_mp4 = os.path.join(out_dir, f'{key}.mp4')
        reshape_pool = reshapePool3d()
        np_results = render_nlf_as_images(smpl_data, motion_indices=list(range(65)), reshape_pool=reshape_pool)

