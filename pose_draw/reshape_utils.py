import numpy as np
import random

def pose_offset(offset_x, offset_y, pose):
    for i in range(len(pose["bodies"]["candidate"])):
        bodies = pose["bodies"]
        # faces = pose["faces"][i:i+1]
        faces = pose["faces"][i]
        # hands = pose["hands"][2*i:2*i+2]
        right_hand = pose["hands"][2 * i]
        left_hand = pose["hands"][2 * i + 1]
        candidate = bodies["candidate"][i]
        subset = bodies["subset"][i]

        for face_point in faces:
            if face_point[0] == -1 and face_point[1] == -1:
                continue
            face_point[0] = face_point[0] + offset_x
            face_point[1] = face_point[1] + offset_y
        for left_hand_point in left_hand:
            if left_hand_point[0] == -1 and left_hand_point[1] == -1:
                continue
            left_hand_point[0] = left_hand_point[0] + offset_x
            left_hand_point[1] = left_hand_point[1] + offset_y
        for right_hand_point in right_hand:
            if right_hand_point[0] == -1 and right_hand_point[1] == -1:
                continue
            right_hand_point[0] = right_hand_point[0] + offset_x
            right_hand_point[1] = right_hand_point[1] + offset_y

        assert len(candidate) == len(subset), f"candidate, length: {len(candidate)} and subset, length: {len(subset)} must have the same length"
        for idx, body_point in enumerate(candidate):
            if subset[idx] == -1:
                continue
            if body_point[0] == -1 and body_point[1] == -1:
                continue
            body_point[0] = body_point[0] + offset_x
            body_point[1] = body_point[1] + offset_y


def pose_whole_scale(scale_x, scale_y, pose):
    # 获取最左端的x 最右端的x 最上端的y 最下端的y
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    for i in range(len(pose["bodies"]["candidate"])):
        bodies = pose["bodies"]
        # faces = pose["faces"][i:i+1]
        faces = pose["faces"][i]
        # hands = pose["hands"][2*i:2*i+2]
        right_hand = pose["hands"][2 * i]
        left_hand = pose["hands"][2 * i + 1]
        candidate = bodies["candidate"][i]
        subset = bodies["subset"][i]

        for face_point in faces:
            if face_point[0] == -1 and face_point[1] == -1:
                continue
            min_x = min(min_x, face_point[0])
            max_x = max(max_x, face_point[0])
            min_y = min(min_y, face_point[1])
            max_y = max(max_y, face_point[1])
        for left_hand_point in left_hand:
            if left_hand_point[0] == -1 and left_hand_point[1] == -1:
                continue
            min_x = min(min_x, left_hand_point[0])
            max_x = max(max_x, left_hand_point[0])
            min_y = min(min_y, left_hand_point[1])
            max_y = max(max_y, left_hand_point[1])
        for right_hand_point in right_hand:
            if right_hand_point[0] == -1 and right_hand_point[1] == -1:
                continue
            min_x = min(min_x, right_hand_point[0])
            max_x = max(max_x, right_hand_point[0])
            min_y = min(min_y, right_hand_point[1])
            max_y = max(max_y, right_hand_point[1])
        assert len(candidate) == len(subset), "candidate and subset must have the same length"
        for idx, body_point in enumerate(candidate):
            if subset[idx] == -1:
                continue
            if body_point[0] == -1 and body_point[1] == -1:
                continue
            min_x = min(min_x, body_point[0])
            max_x = max(max_x, body_point[0])
            min_y = min(min_y, body_point[1])
            max_y = max(max_y, body_point[1])

    # 计算缩放比例
    # 根据bbox中心进行dilate, 倍数为x_scale, y_scale
    bbox_center_x = (min_x + max_x) / 2
    bbox_center_y = (min_y + max_y) / 2
    for i in range(len(pose["bodies"]["candidate"])):
        bodies = pose["bodies"]
        candidate = bodies["candidate"][i]
        for body_point in candidate:
            if body_point[0] == -1 and body_point[1] == -1:
                continue
            body_point[0] = body_point[0] + (body_point[0] - bbox_center_x) * scale_x   # scale越大，越远离中心，变化越大
            body_point[1] = body_point[1] + (body_point[1] - bbox_center_y) * scale_y
        for face_point in faces:
            if face_point[0] == -1 and face_point[1] == -1:
                continue
            face_point[0] = face_point[0] + (face_point[0] - bbox_center_x) * scale_x
            face_point[1] = face_point[1] + (face_point[1] - bbox_center_y) * scale_y
        for left_hand_point in left_hand:
            if left_hand_point[0] == -1 and left_hand_point[1] == -1:
                continue
            left_hand_point[0] = left_hand_point[0] + (left_hand_point[0] - bbox_center_x) * scale_x
            left_hand_point[1] = left_hand_point[1] + (left_hand_point[1] - bbox_center_y) * scale_y
        for right_hand_point in right_hand:
            if right_hand_point[0] == -1 and right_hand_point[1] == -1:
                continue
            right_hand_point[0] = right_hand_point[0] + (right_hand_point[0] - bbox_center_x) * scale_x
            right_hand_point[1] = right_hand_point[1] + (right_hand_point[1] - bbox_center_y) * scale_y


# 如果做多人增强，需要修改逻辑，每个人都有随机性，现在都是做单人增强
def pose_reshape(alpha, pose, 
                 anchor_point, anchor_part, end_point, end_part, 
                 affected_body_points, affected_faces_points, affected_left_hands_points, affected_right_hands_points):
    """
    直接修改原始pose中的某些点的位置，通过根据一个不动点和端点之间的向量偏移被影响点的位置。
    一次只能有一个anchor_point, 一个end_point，多个affected_points
    alpha: 变化比例
    anchor_point: 起始点
    anchor_part: 起始点所属的身体部分，0代表body candidate, 1代表face, 2代表hand
    end_point: 中止点
    end_part: 中止点所属的身体部分，0代表body candidate, 1代表face, 2代表hand
    """
    for i in range(len(pose["bodies"]["candidate"])):
        bodies = pose["bodies"]
        # faces = pose["faces"][i:i+1]
        faces = pose["faces"][i]
        # hands = pose["hands"][2*i:2*i+2]
        right_hand = pose["hands"][2 * i]
        left_hand = pose["hands"][2 * i + 1]
        candidate = bodies["candidate"][i]
        # subset = bodies["subset"][i:i+1]
        subset = bodies["subset"][i]
        assert anchor_part == 0, "anchor part must belong to the body"
        anchor_x, anchor_y = candidate[anchor_point]
        if subset[anchor_point] == -1:
            return

        if end_part == 0:
            end_x, end_y = candidate[end_point]
            if subset[end_point] == -1:
                return
        elif end_part == 1:
            end_x, end_y = faces[end_point]
        # 不考虑Hands
        if end_x == -1 and end_y == -1:
            return
        
        vector_x = end_x - anchor_x
        vector_y = end_y - anchor_y
        offset_x = vector_x * alpha
        offset_y = vector_y * alpha
        
        for affected_body_point in affected_body_points:
            if subset[affected_body_point] == -1:
                continue
            affected_x, affected_y = candidate[affected_body_point]
            candidate[affected_body_point] = [affected_x + offset_x, affected_y + offset_y]
        for affected_faces_point in affected_faces_points:
            affected_x, affected_y = faces[affected_faces_point]
            if affected_x == -1 and affected_y == -1:
                continue
            faces[affected_faces_point] = [affected_x + offset_x, affected_y + offset_y]
        for affected_hands_point in affected_left_hands_points:
            affected_x, affected_y = left_hand[affected_hands_point]
            if affected_x == -1 and affected_y == -1:
                continue
            left_hand[affected_hands_point] = [affected_x + offset_x, affected_y + offset_y]
        for affected_hands_point in affected_right_hands_points:
            affected_x, affected_y = right_hand[affected_hands_point]
            if affected_x == -1 and affected_y == -1:
                continue
            right_hand[affected_hands_point] = [affected_x + offset_x, affected_y + offset_y]

class reshapePool:
    def __init__(self, alpha):  # 对每个视频只初始化一次
        self.faces_indices = np.arange(0, 68)
        self.left_hands_indices = np.arange(0, 21)
        self.right_hands_indices = np.arange(0, 21)
        self.alpha = alpha  # 0.1
        self.offset_x = random.uniform(-1/64, 1/64)
        self.offset_y = random.uniform(-1/64, 1/64)
        self.scale_x = random.uniform(-0.1, 0.1)
        self.scale_y = random.uniform(-0.1, 0.1)
        self.reshape_methods = [
            self.extend_body,
            self.extend_arm,
            self.extend_leg,
            self.shrink_body,
            self.shrink_arm,
            self.shrink_leg,
            self.offset_wholebody,
            self.scale_wholebody,
        ]
        self.selected_methods = random.sample(self.reshape_methods, 2)
        # self.selected_methods = [self.reshape_face]

    def apply_random_reshapes(self, pose):
        # Apply the two selected reshape methods
        for method in self.selected_methods:
            method(pose)

    def offset_wholebody(self, pose):
        pose_offset(self.offset_x, self.offset_y, pose)

    def scale_wholebody(self, pose):
        pose_whole_scale(self.scale_x, self.scale_y, pose)


    def extend_body(self, pose):
        pose_reshape(self.alpha, pose,
                     1, 0, 8, 0,
                     [8, 9, 10], [], [], []
                     )
        pose_reshape(self.alpha, pose,
                    1, 0, 11, 0,
                    [11, 12, 13], [], [], []
                    )
        
    def shrink_body(self, pose):
        pose_reshape(-self.alpha, pose,
                     1, 0, 8, 0,
                     [8, 9, 10], [], [], []
                     )
        pose_reshape(-self.alpha, pose,
                    1, 0, 11, 0,
                    [11, 12, 13], [], [], []
                    )
        
    def extend_arm(self, pose):
        pose_reshape(self.alpha, pose,
                     2, 0, 3, 0,
                     [3, 4], [], self.left_hands_indices, []
                     )
        pose_reshape(self.alpha, pose,
                    5, 0, 6, 0,
                    [6, 7], [], [], self.right_hands_indices
                    )
        pose_reshape(self.alpha, pose,
                     3, 0, 4, 0,
                     [4], [], self.left_hands_indices, []
                     )
        pose_reshape(self.alpha, pose,
                    6, 0, 7, 0,
                    [7], [], [], self.right_hands_indices
                    )
            
    def shrink_arm(self, pose):
        pose_reshape(-self.alpha, pose,
                     2, 0, 3, 0,
                     [3, 4], [], self.left_hands_indices, []
                     )
        pose_reshape(-self.alpha, pose,
                    5, 0, 6, 0,
                    [6, 7], [], [], self.right_hands_indices
                    )
        pose_reshape(-self.alpha, pose,
                     3, 0, 4, 0,
                     [4], [], self.left_hands_indices, []
                     )
        pose_reshape(-self.alpha, pose,
                    6, 0, 7, 0,
                    [7], [], [], self.right_hands_indices
                    )

    def extend_leg(self, pose):
        pose_reshape(self.alpha, pose,
                     8, 0, 9, 0,
                     [9, 10], [], [], []
                     )
        pose_reshape(self.alpha, pose,
                    11, 0, 12, 0,
                    [12, 13], [], [], []
                    )
        pose_reshape(self.alpha, pose,
                     9, 0, 10, 0,
                     [10], [], [], []
                     )
        pose_reshape(self.alpha, pose,
                    12, 0, 13, 0,
                    [13], [], [], []
                    )
        
    def shrink_leg(self, pose):
        pose_reshape(-self.alpha, pose,
                     8, 0, 9, 0,
                     [9, 10], [], [], []
                     )
        pose_reshape(-self.alpha, pose,
                    11, 0, 12, 0,
                    [12, 13], [], [], []
                    )
        pose_reshape(-self.alpha, pose,
                     9, 0, 10, 0,
                     [10], [], [], []
                     )
        pose_reshape(-self.alpha, pose,
                    12, 0, 13, 0,
                    [13], [], [], []
                    )

    def extend_shoulder(self, pose):
        pose_reshape(self.alpha, pose,
                     1, 0, 2, 0,
                     [2, 3, 4], [], self.left_hands_indices, []
                     )
        pose_reshape(self.alpha, pose,
                    1, 0, 5, 0,
                    [5, 6, 7], [], [], self.right_hands_indices
                    )
        
    def shrink_shoulder(self, pose):
        pose_reshape(-self.alpha, pose,
                     1, 0, 2, 0,
                     [2, 3, 4], [], self.left_hands_indices, []
                     )
        pose_reshape(-self.alpha, pose,
                    1, 0, 5, 0,
                    [5, 6, 7], [], [], self.right_hands_indices
                    )
        
    
    def dilate_face(self, pose):
        for i in self.faces_indices:
            pose_reshape(self.alpha, pose,
                     1, 0, i, 1,
                     [], [i], [], []
                     )
        
    def shrink_face(self, pose):
        for i in self.faces_indices:
            pose_reshape(-self.alpha, pose,
                     1, 0, i, 1,
                     [], [i], [], []
                     )