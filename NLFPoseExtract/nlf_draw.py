import cv2
import numpy as np
import math
from PIL import Image
from DWPoseProcess.dwpose.util import draw_bodypose


def process_data_to_COCO_format(joints):
    """Args:
        joints: numpy array of shape (24, 2) or (24, 3)
    Returns:
        new_joints: numpy array of shape (17, 2) or (17, 3)
    """
    if joints.ndim != 2:
        raise ValueError(f"Expected shape (24,2) or (24,3), got {joints.shape}")

    dim = joints.shape[1]  # 2D or 3D

    mapping = {
        15: 0,   # head
        12: 1,   # neck
        17: 2,   # left shoulder
        16: 5,   # right shoulder
        19: 3,   # left elbow
        18: 6,   # right elbow
        21: 4,   # left hand
        20: 7,   # right hand
        2: 8,    # left pelvis
        1: 11,   # right pelvis
        5: 9,    # left knee
        4: 12,   # right knee
        8: 10,   # left feet
        7: 13,   # right feet
    }

    new_joints = np.zeros((18, dim), dtype=joints.dtype)
    for src, dst in mapping.items():
        new_joints[dst] = joints[src]

    return new_joints


def intrinsic_matrix_from_field_of_view(imshape, fov_degrees:float =55):   # nlf default fov_degrees 55
    imshape = np.array(imshape)
    fov_radians = fov_degrees * np.array(np.pi / 180)
    larger_side = np.max(imshape)
    focal_length = larger_side / (np.tan(fov_radians / 2) * 2)
    # intrinsic_matrix 3*3
    return np.array([   
        [focal_length, 0, imshape[1] / 2],
        [0, focal_length, imshape[0] / 2],
        [0, 0, 1],
    ])


def p3d_to_p2d(point_3d, height, width):    # point3d n*num_points*3
    camera_matrix = intrinsic_matrix_from_field_of_view((height,width))
    camera_matrix = np.expand_dims(camera_matrix, axis=0)
    camera_matrix = np.expand_dims(camera_matrix, axis=0)    # 1*1*3*3
    point_3d = np.expand_dims(point_3d,axis=-1)     # n*num_points*3*1
    point_2d = (camera_matrix@point_3d).squeeze(-1)  # n*num_points*3
    point_2d[:,:,:2] = point_2d[:,:,:2]/point_2d[:,:,2:3]  # 相对位置
    return point_2d[:,:,:]      # n*num_points*2

def preview_nlf_2d(data):
    """ return a list of images """
    height, width = data['video_height'], data['video_width']
    offset = [height, width, 0]
    np_images = []
    for image_result in data['pose']['joints3d_nonparam']:
        final_canvas = np.zeros(shape=(offset[0], offset[1], 3), dtype=np.uint8)
        for joints3d in image_result:  # 每个人的pose
            canvas = np.zeros(shape=(offset[0], offset[1], 3), dtype=np.uint8)
            joints3d = joints3d.cpu().numpy()
            joints2d = p3d_to_p2d(joints3d, offset[0], offset[1])
            joints2d = joints2d[0][:, :2] 
            joints2d[:, 0] = joints2d[:, 0] / offset[1]  # x坐标归一化
            joints2d[:, 1] = joints2d[:, 1] / offset[0]  # y坐标归一化
            joints2d = process_data_to_COCO_format(joints2d)
            subset = np.expand_dims(np.concatenate([np.arange(14), [-1, -1, -1, -1]]), axis=0)
            canvas = draw_bodypose(canvas, joints2d, subset)
            final_canvas = final_canvas + canvas
        np_images.append(final_canvas)

    return np_images