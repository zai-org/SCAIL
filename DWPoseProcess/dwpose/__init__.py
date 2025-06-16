# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody


class DWposeDetector:
    def __init__(self, use_batch=False):
        self.use_batch = use_batch
        pass

    def to(self, device):
        self.pose_estimation = Wholebody(device, self.use_batch)
        return self

    def _get_multi_result_from_est(self, candidate, subset, det_result, H, W):
        nums, keys, locs = candidate.shape  # n 所有身体关键点数量，坐标
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        score = subset[:, :24]              # 前18个是躯干骨骼  score(n, 18)
        body_candidate = candidate[:, :24].copy()     # body(n, 24, 2)
        body_score = copy.deepcopy(score)       # 已经去过max_ind
        for i in range(len(score)):  # n 个
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = j
                else:
                    score[i][j] = -1    # 躯干中去除掉不可见的骨骼

        un_visible = subset < 0.3       
        candidate[un_visible] = -1      # 全部关键点中去掉不可见骨骼

        faces = candidate[:, 24:92]

        hands = candidate[:, 92:113]    # hands(2*n, 21, 2)
        hands = np.vstack([hands, candidate[:, 113:]])  # 

        bodies = dict(candidate=body_candidate, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        new_det_result = []
        for bbox in det_result:
            x1, y1, x2, y2 = bbox
            new_x1 = x1 / W
            new_y1 = y1 / H
            new_x2 = x2 / W
            new_y2 = y2 / H
            new_bbox = [new_x1, new_y1, new_x2, new_y2]
            new_det_result.append(new_bbox)

        return pose, body_score, new_det_result     # body_score是原始的躯干骨骼分数

    # def _get_result_from_est(self, input_image, candidate, subset, det_result, image_resolution, output_type, H, W):
    #     nums, keys, locs = candidate.shape
    #     candidate[..., 0] /= float(W)
    #     candidate[..., 1] /= float(H)
    #     score = subset[:, :18]              # 前18个是躯干骨骼  score(n, 18)
    #     max_ind = np.mean(score, axis=-1).argmax(axis=0)    # 返回分数最高的锚框对应的骨骼
    #     score = score[[max_ind]]
    #     body = candidate[:, :18].copy()
    #     body = body[[max_ind]]
    #     nums = 1
    #     body = body.reshape(nums * 18, locs)    # Moore-AA只有一个人体, 0-18表示body
    #     body_score = copy.deepcopy(score)       # 已经去过max_ind
    #     for i in range(len(score)):
    #         for j in range(len(score[i])):
    #             if score[i][j] > 0.3:
    #                 score[i][j] = int(18 * i + j)
    #             else:
    #                 score[i][j] = -1    # 躯干中去除掉不可见的骨骼

    #     un_visible = subset < 0.3       
    #     candidate[un_visible] = -1      # 全部关键点中去掉不可见骨骼

    #     foot = candidate[:, 18:24]

    #     faces = candidate[[max_ind], 24:92]

    #     hands = candidate[[max_ind], 92:113]
    #     hands = np.vstack([hands, candidate[[max_ind], 113:]])

    #     bodies = dict(candidate=body, subset=score)
    #     pose = dict(bodies=bodies, hands=hands, faces=faces)

    #     return pose, body_score, det_result     # body_score是原始的躯干骨骼分数

    def __call__(
        self,
        input,
        **kwargs,
    ):                          
        if not self.use_batch:
            # PIL要不要颜色反转？
            input = cv2.cvtColor(
                np.array(input, dtype=np.uint8), cv2.COLOR_RGB2BGR
            )
            input = HWC3(input)
            H, W, C = input.shape

            with torch.no_grad():
                candidate, subset, det_result = self.pose_estimation(input)   # candidate (n, 134, 2) 候选点 / subset (n, 134) 得分
                return self._get_multi_result_from_est(candidate, subset, det_result, H, W)
        else:
            raise NotImplementedError("DWposeDetector does not support batch mode")





            

