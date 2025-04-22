def convert_3dpose_to_2dpose_body(body_keypoints, face_keypoints):
    """
    将20点的3D坐标映射到18点的2D坐标。
    :param poses: 输入的20点坐标列表，每个点为 [x, y, z]
    :return: 映射得到的18点坐标列表，每个点为 [x, y]
    """
    # 映射关系：索引位置
    body_mapping = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 8: 8, 9: 9, 10: 10,
        14: 11, 15: 12, 16: 13, 11: 18, 12: 19, 13: 20, 17: 21, 18: 22, 19: 23
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
            condidate = convert_3dpose_to_2dpose_body(obj["body"], obj["face"])
            subset = [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
        13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.]]
            faces = [convert_3dpose_to_2dpose_face(obj["face"])]
            hands = convert_3dpose_to_2dpose_hand(obj["left_hand"],obj["right_hand"], obj["body"])
            poses.append({"bodies":{"candidate": condidate, "subset": subset}, "faces": faces, "hands": hands})
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
    

