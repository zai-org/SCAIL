import numpy as np

def convert_3dpose_to_2dpose_body(body_keypoints, face_keypoints):
    """
    将20点的3D坐标映射到18点的2D坐标。
    :param poses: 输入的20点坐标列表，每个点为 [x, y, z]
    :return: 映射得到的18点坐标列表，每个点为 [x, y]
    """
    # 映射关系：索引位置
    body_mapping = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 8, 9: 9, 10: 10, 11: 23, 13: 22, 12: 21,
        14: 11, 15: 12, 16: 13, 17: 20, 18: 18, 19: 19
    }
    face_mapping = {
        1: 16, 8: 14, 4: 0, 7: 15, 0: 17
    }
    
    # 初始化18点坐标列表，默认值为 [-1, -1]
    result = [[-1, -1] for _ in range(24)]
    
    # 遍历映射关系，将对应的20点坐标映射到18点坐标
    for src_idx, dst_idx in body_mapping.items():
        if src_idx < len(body_keypoints):  # 确保索引不越界
            result[dst_idx] = [body_keypoints[src_idx][1],body_keypoints[src_idx][0]]   # 提取 x, y 坐标
    for src_idx, dst_idx in face_mapping.items():
        if src_idx < len(face_keypoints):
            result[dst_idx] = [face_keypoints[src_idx][1], face_keypoints[src_idx][0]]
    return result

def convert_3dpose_to_2dpose_hand(left_hand_keypoints, right_hand_keypoints, body_keypoints):
    """
    将20点的3D坐标映射到18点的2D坐标。
    :param poses: 输入的20点坐标列表，每个点为 [x, y, z]
    :return: 映射得到的18点坐标列表，每个点为 [x, y]
    """
    # 映射关系：索引位置
    hand_mapping = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
        10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18,
        18: 19, 19: 20
    }

    body_mapping_left = {3: 0}
    body_mapping_right = {6: 0}
    
    # 初始化18点坐标列表，默认值为 [-1, -1]
    left_result = [[-1, -1] for _ in range(21)]
    right_result = [[-1, -1] for _ in range(21)]
    
    # 遍历映射关系，将对应的20点坐标映射到18点坐标
    for src_idx, dst_idx in hand_mapping.items():
        if src_idx < len(left_hand_keypoints):  # 确保索引不越界
            left_result[dst_idx] = [left_hand_keypoints[src_idx][1], left_hand_keypoints[src_idx][0]]  # 提取 x, y 坐标
            right_result[dst_idx] = [right_hand_keypoints[src_idx][1], right_hand_keypoints[src_idx][0]]
    
    for src_idx, dst_idx in body_mapping_left.items():
        if src_idx < len(body_keypoints):
            left_result[dst_idx] = [body_keypoints[src_idx][1], body_keypoints[src_idx][0]]
    for src_idx, dst_idx in body_mapping_right.items():
        if src_idx < len(body_keypoints):
            right_result[dst_idx] = [body_keypoints[src_idx][1], body_keypoints[src_idx][0]]
    
    return [left_result, right_result]

def convert_3dpose_to_2dpose_face(face_keypoints):
    result = [[-1, -1] if i in [0, 1, 4, 5, 6, 7, 8] else [pt[1], pt[0]] for i, pt in enumerate(face_keypoints)]
    return result

def read_pose_from_jsonl(jsonl_path):
    import jsonlines
    poses = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            breakpoint()
            body_points = convert_3dpose_to_2dpose_body(obj["body"], obj["face"])
            candidate = [body_points]
            subset = [[i for i in range(24)]]
            faces = [convert_3dpose_to_2dpose_face(obj["face"])]
            hands = convert_3dpose_to_2dpose_hand(obj["left_hand"],obj["right_hand"], obj["body"])
            poses.append({"bodies":{"candidate": candidate, "subset": subset}, "faces": faces, "hands": hands})
    return poses


def mix_3d_poses(poses_dwpose, poses_3dpose):
    '''
    组合两种pose，用3dPose的身体，DWPose的face和hand
    '''
    poses = []
    for pose_dwpose, pose_3dpose in zip(poses_dwpose, poses_3dpose):
        pose = {
            "bodies": {
                "candidate": pose_3dpose["bodies"]["candidate"],
                "subset": pose_dwpose["bodies"]["subset"]
            },  
            "faces": pose_dwpose["faces"],
            "hands": pose_dwpose["hands"]
        }
        poses.append(pose)
    return poses

def correct_hand_from_3d(hand_keypoints_dwpose, hand_keypoints_3dpose):
    '''
    如果dwpose的手部关节点和3dpose的手部关节点相差过大，则去掉最远的那一端
    '''
    edges_palm = [
        [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8],
        [9, 10], [10, 11], [11, 12],
        [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20],
    ]
    edges_finger = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17]]
    max_length_palm = 0
    max_length_finger = 0
    for edge in edges_palm:
        limb_length_3dpose = np.linalg.norm(np.array(hand_keypoints_3dpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[1]]))
        if limb_length_3dpose > max_length_palm:
            max_length_palm = limb_length_3dpose
    for edge in edges_finger:
        limb_length_3dpose = np.linalg.norm(np.array(hand_keypoints_3dpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[1]]))
        if limb_length_3dpose > max_length_finger:
            max_length_finger = limb_length_3dpose
    for edge in edges_palm:
        limb_length_dwpose = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_dwpose[edge[1]]))
        if limb_length_dwpose > max_length_palm * 1.5:
            if -1 in hand_keypoints_dwpose[edge[0]] or -1 in hand_keypoints_dwpose[edge[1]] or -1 in hand_keypoints_3dpose[edge[0]] or -1 in hand_keypoints_3dpose[edge[1]]:
                continue
            distance_point_0 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[0]]))
            distance_point_1 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[1]]) - np.array(hand_keypoints_3dpose[edge[1]]))
            if distance_point_0 > distance_point_1:
                hand_keypoints_dwpose[edge[1]] = [-1, -1]
            else:
                hand_keypoints_dwpose[edge[0]] = [-1, -1]
    for edge in edges_finger:
        limb_length_dwpose = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_dwpose[edge[1]]))
        if limb_length_dwpose > max_length_finger * 1.5:
            if -1 in hand_keypoints_dwpose[edge[0]] or -1 in hand_keypoints_dwpose[edge[1]] or -1 in hand_keypoints_3dpose[edge[0]] or -1 in hand_keypoints_3dpose[edge[1]]:
                continue
            distance_point_0 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[0]]) - np.array(hand_keypoints_3dpose[edge[0]]))
            distance_point_1 = np.linalg.norm(np.array(hand_keypoints_dwpose[edge[1]]) - np.array(hand_keypoints_3dpose[edge[1]]))
            if distance_point_0 > distance_point_1:
                hand_keypoints_dwpose[edge[1]] = [-1, -1]
            else:
                hand_keypoints_dwpose[edge[0]] = [-1, -1]
    return hand_keypoints_dwpose

def correct_body_from_3d(body_keypoints_dwpose, body_keypoints_3dpose, subset_dwpose, subset_3dpose):
    '''
    如果dwpose的骨骼长度和3dpose的骨骼长度相差过大，则去掉最远的那一端
    '''
    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]
        
    for ori_limb in limbSeq:
        limb = [ori_limb[0] - 1, ori_limb[1] - 1]
        limb_length_dwpose = np.linalg.norm(np.array(body_keypoints_dwpose[limb[0]]) - np.array(body_keypoints_dwpose[limb[1]]))
        limb_length_3dpose = np.linalg.norm(np.array(body_keypoints_3dpose[limb[0]]) - np.array(body_keypoints_3dpose[limb[1]]))
        if subset_dwpose[0][limb[0]] == -1 or subset_dwpose[0][limb[1]] == -1 or subset_3dpose[0][limb[0]] == -1 or subset_3dpose[0][limb[1]] == -1:
            continue
        if limb_length_dwpose > limb_length_3dpose * 2:
            # 判断较远端
            distance_point_0 = np.linalg.norm(np.array(body_keypoints_dwpose[limb[0]]) - np.array(body_keypoints_3dpose[limb[0]]))
            distance_point_1 = np.linalg.norm(np.array(body_keypoints_dwpose[limb[1]]) - np.array(body_keypoints_3dpose[limb[1]]))
            if distance_point_0 > distance_point_1:
                if limb[1] == 1:    # 核心
                    continue
                body_keypoints_dwpose[limb[1]] = [-1, -1]
                subset_dwpose[0][limb[1]] = -1
            else:
                if limb[0] == 1:    # 核心
                    continue
                body_keypoints_dwpose[limb[0]] = [-1, -1]
                subset_dwpose[0][limb[0]] = -1
    return body_keypoints_dwpose, subset_dwpose

def correct_full_pose_from_3d(poses_dwpose, poses_3dpose):
    '''
    如果dwpose的骨骼长度和3dpose的骨骼长度相差过大，则去掉离3d pose最远的那一端
    '''
    poses = []
    for pose_dwpose, pose_3dpose in zip(poses_dwpose, poses_3dpose):
        new_candidate, new_subset = correct_body_from_3d(pose_dwpose["bodies"]["candidate"], pose_3dpose["bodies"]["candidate"], pose_dwpose["bodies"]["subset"], pose_3dpose["bodies"]["subset"])
        new_hands_0 = correct_hand_from_3d(pose_dwpose["hands"][0], pose_3dpose["hands"][0])
        new_hands_1 = correct_hand_from_3d(pose_dwpose["hands"][1], pose_3dpose["hands"][1])
        pose = {
            "bodies": {
                "candidate": new_candidate,
                "subset": new_subset
            },  
            "faces": pose_dwpose["faces"],
            "hands": [new_hands_0, new_hands_1]
        }
        poses.append(pose)

    return poses