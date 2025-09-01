import os
import random
from pathlib import Path
import multiprocessing
import numpy as np
import time
from dwpose import DWposeDetector
from DWPoseProcess.AAUtils import save_videos_from_pil
from collections import deque
import shutil
import torch
import yaml
from pose_draw.draw_pose_main import draw_pose_to_canvas
from extractUtils import check_single_human_requirements, check_multi_human_requirements, human_select
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, TimeoutError, as_completed
from AAUtils import read_frames
from decord import VideoReader
from fractions import Fraction
import io
import gc
from PIL import Image
from multiprocessing import Process
import decord
import json
import glob
import sys
from extract_dwpose import convert_scores_to_specific_bboxes, get_bbox_from_position_list
from extract_mp4_dwpose import draw_bbox_to_mp4

hands_bbox_path = '/workspace/ywh_data/DataProcessNew/pexels1k/hands/0a0abf80a9c8c500250cde96612a9b12.pt'
faces_bbox_path = '/workspace/ywh_data/DataProcessNew/pexels1k/faces/0a0abf80a9c8c500250cde96612a9b12.pt'
mp4_path = hands_bbox_path.replace('.pt', '.mp4').replace('hands', 'dwpose')
pil_frames = read_frames(mp4_path)

hands_bboxes = torch.load(hands_bbox_path)
faces_bboxes = torch.load(faces_bbox_path)

preview_results = draw_bbox_to_mp4(pil_frames, hands_bboxes)
preview_results = draw_bbox_to_mp4(preview_results, faces_bboxes)
save_videos_from_pil(preview_results, '/workspace/yanwenhao/dwpose_draw/preview_results.mp4', fps=16)




