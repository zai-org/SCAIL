import os
import sys
import cv2
import torch
import pickle
import torchvision
import shutil
import glob
from tqdm import tqdm   
import decord
from decord import VideoReader, cpu, gpu
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
from NLFPoseExtract.nlf_render import render_nlf_as_images
from DWPoseProcess.dwpose import DWposeDetector
from DWPoseProcess.AAUtils import save_videos_from_pil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from DWPoseProcess.extract_nlfpose import process_video_nlf

def process_video_dwpose(model, frames, height, width):

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


# 对3d做reshape的时候，目前只做reshape，移动按照2d? 按照3d移动一些看看

if __name__ == '__main__':
    model = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()

    evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"
    decord.bridge.set_bridge("torch")


    for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
        ori_video_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        out_path_mp4 = os.path.join(evaluation_dir, subdir, 'smpl_render.mp4')
        vr = VideoReader(ori_video_path)
        frames = vr.get_batch(list(range(len(vr))))
        frames = torch.from_numpy(frames) if type(frames) is not torch.Tensor else frames
        height = frames.shape[1]
        width = frames.shape[2]
        length = frames.shape[0]

        output_meta = process_video_nlf(model, frames, height, width)
        dwpose_meta = process_video_dwpose(model, frames, height, width)
        pil_results = render_nlf_as_images(output_meta, motion_indices=list(range(length)), output_path=out_path_mp4)

