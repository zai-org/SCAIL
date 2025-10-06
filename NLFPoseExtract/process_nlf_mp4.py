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
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy



if __name__ == '__main__':
    model_nlf = torch.jit.load("/workspace/yanwenhao/dwpose_draw/NLFPoseExtract/nlf_l_multi_0.3.2.torchscript").cuda().eval()

    evaluation_dir = "/workspace/ywh_data/EvalSelf/evaluation_300_old"
    decord.bridge.set_bridge("torch")

    for subdir_idx, subdir in tqdm(enumerate(sorted(os.listdir(evaluation_dir)))):
        mp4_path = os.path.join(evaluation_dir, subdir, 'GT.mp4')
        out_path = os.path.join(evaluation_dir, subdir, 'smpl_render.mp4')
        vr = VideoReader(mp4_path)
        vr_frames = vr.get_batch(list(range(len(vr))))
        height, width = vr_frames.shape[1], vr_frames.shape[2]
        nlf_results = process_video_nlf(model_nlf, vr_frames, height, width)
        frames_np = render_nlf_as_images(nlf_results, list(range(len(vr_frames))))
        print("save video to ", out_path)
        mpy.ImageSequenceClip(frames_np, fps=16).write_videofile(out_path)

