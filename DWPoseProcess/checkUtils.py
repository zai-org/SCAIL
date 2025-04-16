import numpy as np
from collections import deque
import numpy as np
from collections import deque
import math

##############################锚框相关################################
def check_consistant(boxA, boxB, scoreA_lst, scoreB_lst, beta, all_threshold):
    """
    计算两个锚框之间的 IoU（交并比）以及分数的变化比例，来判断连续性
    """
    # 计算交集框的坐标
    scoreA = scoreA_lst[0]
    scoreB = scoreB_lst[0]
    iou = get_IoU(boxA, boxB)

    reduction_ratio = (scoreA - scoreB) / (scoreA + scoreB) if scoreA > scoreB else 0  # 如果分数减少的很多，就更不连续

    return iou - reduction_ratio * beta > all_threshold


def get_IoU(boxA, boxB):
    """
    计算两个锚框之间的 IoU（交并比）以及分数的变化比例，来判断连续性
    """
    # 计算交集框的坐标
    x1_int = max(boxA[0], boxB[0])
    y1_int = max(boxA[1], boxB[1])
    x2_int = min(boxA[2], boxB[2])
    y2_int = min(boxA[3], boxB[3])

    # 计算交集的面积
    inter_width = max(0, x2_int - x1_int)
    inter_height = max(0, y2_int - y1_int)
    inter_area = inter_width * inter_height

    # 计算两个锚框的面积
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集的面积
    union_area = areaA + areaB - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou

def check_bbox_ok_for_video(bbox, reference_width, reference_height):
    """
    每一帧都要检查，判断 bbox 是否符合视频格式下的要求
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # 防止非法 bbox
    if bbox_width <= 0 or bbox_height <= 0:
        return False

    bbox_area = bbox_width * bbox_height
    video_area = reference_width * reference_height
    aspect_ok = bbox_height > bbox_width * 0.5
    if not aspect_ok:
        # print("filtered: bbox aspect not like human")
        return False

    if reference_width > reference_height:
        # 横屏
        min_bbox_area = video_area * (1 / 40)
        if bbox_area < min_bbox_area:
            # print("filtered: bbox too small")
            return False
        max_bbox_width = reference_width * (2 / 3)
        if bbox_width > max_bbox_width:
            # print("filtered: bbox too wide")
            return False
        center_x = (x1 + x2) / 2
        left_limit = reference_width * (1 / 5)
        right_limit = reference_width * (4 / 5)
        if not (left_limit <= center_x <= right_limit):
            # print("filtered: bbox not in center")
            return False

    else:
        # 竖屏
        max_bbox_area = video_area * (4 / 5)
        if bbox_area > max_bbox_area:
            # print("filtered: bbox too large")
            return False

    return True



##############################判断点是否满足################################

def part5_valid(valid_joints):
    """
    判断身体五块是不是都有点
    
    参数:
    valid_joints:布尔数组
    
    返回:
    bool: 如果满足要求返回True，否则返回False。
    """

    ###  认为以下的视频是满足我们采样需要的：
    ###  A只有上半身手部动作的：手部有可能在挥舞过程中移出屏幕，但是上半身应该一直在屏幕内，此时1-0, 1-2, 1-5这三条骨骼应该都存在；14-17应该至少有一个点存在
    ###  B全身动作：1-0, 1-2, 1-5, 1-8, 1-11这五条骨骼应该都存在
    
    top_core_joints = valid_joints[[0,1,2,5]]
    top_nece_joints = valid_joints[[14,15,16,17]]
    if all(top_core_joints) and any(top_nece_joints):
        return True
    
    wholebody_core_joints = valid_joints[[0,1,2,5,8,11]]
    if all(wholebody_core_joints):
        return True
    return False
    
    
def check_valid_sequence(valid_keypoints, threshold=0.3):
    valid_joints = np.zeros(18)
    for valid_keypoint in valid_keypoints:
        valid_joints += valid_keypoint
    return part5_valid(valid_joints)


def get_valid_indice_from_keypoints(ref_part_poses, ref_part_indices):
    # ref_part_poses: poses序列，ref_part_indices poses序列里面每个值对应的在整个序列里的index
    # return: 每个pose序列里面，满足要求的pose的index
    valid_indice = []
    for i, (keypoint_all, indice) in enumerate(zip(ref_part_poses, ref_part_indices)):
        body_subset = keypoint_all["bodies"]["subset"][0]
        valid_joints = body_subset > -1 # 得到一个布尔索引

        if not part5_valid(valid_joints):
            continue

        faces = keypoint_all["faces"][0]    

        left_eye = faces[36:42]  # 左眼关键点   5个关键点有4个认为有左眼
        right_eye = faces[42:48]  # 右眼关键点  5个关键点有4个认为有右眼
        nose = faces[27:36]  # 鼻子关键点   8个关键点有5个认为有鼻子
        mouth = faces[48:68]  # 嘴巴关键点  21个关键点有15个认为有嘴巴

        # 计算每个部位有效的关键点数
        left_eye_valid = sum(1 for point in left_eye if point[0] > 0 and point[1] > 0)
        right_eye_valid = sum(1 for point in right_eye if point[0] > 0 and point[1] > 0)
        nose_valid = sum(1 for point in nose if point[0] > 0 and point[1] > 0)
        mouth_valid = sum(1 for point in mouth if point[0] > 0 and point[1] > 0)

        # 如果有两个或以上部位有效，则认为是正脸
        valid_face_parts = 0
        if left_eye_valid >= 4:
            valid_face_parts += 1
        if right_eye_valid >= 4:
            valid_face_parts += 1
        if nose_valid >= 5:
            valid_face_parts += 1
        if mouth_valid >= 15:
            valid_face_parts += 1

        if valid_face_parts >= 2:
            valid_indice.append(indice)
    
    return valid_indice
    
def check_from_keypoints_core_keypoints(keypoints, bboxs, IoU_thresthold, reference_width, reference_height):
    # 用于keypoints版本，根据每一帧的18个keypoints和bbox iou来判断是否满足要求
    valid_sequence = deque(maxlen=4)
    for i, (keypoint_all, bbox_all) in enumerate(zip(keypoints, bboxs)):
        body_subset = keypoint_all["bodies"]["subset"][0]
        valid_joints = body_subset > -1 # 得到一个布尔索引
        if len(valid_sequence) == 4:
            if not check_valid_sequence(valid_sequence):
                # print("filtered: 骨骼不满足要求")
                return False   # 关键点异常
        valid_sequence.append(valid_joints)
    return True


def check_from_keypoints_bbox(keypoints, bboxs, IoU_thresthold, reference_width, reference_height):
    # 用于keypoints版本，根据每一帧的18个keypoints和bbox iou来判断是否满足要求
    last_bbox = None
    for i, (keypoint_all, bbox_all) in enumerate(zip(keypoints, bboxs)):
        if not len(bbox_all):
            return False
        bbox = bbox_all[0]
        if not check_bbox_ok_for_video(bbox, reference_width, reference_height):
            return False   # bbox大小异常
        if last_bbox is not None:
            if not get_IoU(bbox, last_bbox) > IoU_thresthold:
                return False   # IoU异常
        last_bbox = bbox
    return True




def check_from_keypoints_stick_movement(keypoints, bboxs, IoU_thresthold, reference_width, reference_height):
    # 骨骼选择：列表中每个元组表示由两个关节确定一条骨骼：格式 (joint_a, joint_b)
    bones = [(1, 0), (1, 2), (1, 5), (1, 8), (1, 11)]
    
    # 定义阈值
    single_bone_threshold = math.radians(60)   # 60度对应的弧度
    overall_movement_threshold = math.radians(5) # 5度对应的弧度

    total_angle_diff = 0.0
    count = 0

    max_delta_list = []
    # 遍历从第二帧开始，对比前一帧和当前帧
    for i in range(1, len(keypoints)):
        # 获取上一帧和当前帧的关键点数据（格式为 (18,3) 数组）
        prev_frame_subset = keypoints[i-1]["bodies"]["subset"][0]
        curr_frame_subset = keypoints[i]["bodies"]["subset"][0]
        prev_frame_keypoints = keypoints[i-1]["bodies"]["candidate"]
        curr_frame_keypoints = keypoints[i]["bodies"]["candidate"]

        
        max_delta = 0
        for (j1, j2) in bones:
            # 检查上一帧中两个关节是否有效（假设 x, y 坐标需大于 0 才认为有效）
            if prev_frame_subset[j1] < 0 or prev_frame_subset[j2] < 0:
                continue
            if curr_frame_subset[j1] < 0 or curr_frame_subset[j2] < 0:
                continue

            # 计算上一帧和当前帧中对应骨骼的向量（方向一致，均从 j1 指向 j2）
            vec_prev = np.array([prev_frame_keypoints[j2][0] - prev_frame_keypoints[j1][0],
                                 prev_frame_keypoints[j2][1] - prev_frame_keypoints[j1][1]])
            vec_curr = np.array([curr_frame_keypoints[j2][0] - curr_frame_keypoints[j1][0],
                                 curr_frame_keypoints[j2][1] - curr_frame_keypoints[j1][1]])
            
            # 如果向量模长为0，则无法计算角度，跳过
            if np.linalg.norm(vec_prev) == 0 or np.linalg.norm(vec_curr) == 0:
                continue
            # 计算向量对应的角度（弧度制）
            angle_prev = math.atan2(vec_prev[1], vec_prev[0])
            angle_curr = math.atan2(vec_curr[1], vec_curr[0])
            
            # 计算角度差，并规范到 [0, pi] 范围
            delta = abs(angle_curr - angle_prev)
            if delta > math.pi:
                delta = 2 * math.pi - delta
                
            max_delta = max(delta, max_delta)
        max_delta_list.append(max_delta)

    return max_delta_list
    