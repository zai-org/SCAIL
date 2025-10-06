import numpy as np
import random
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_to_p2d
import torch


def offset_2d(offset_x, offset_y, pose):
    for i in range(len(pose["bodies"]["candidate"])):
        bodies = pose["bodies"]
        faces = pose["faces"][i]
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


def pose_reshape_3d(alpha, smpl_joints_list, 
                 anchor_point, end_point, 
                 affected_body_points):
    """
    # joints3d: n, 24, 3
    alpha: 变化比例
    anchor_point: 起始点
    anchor_part: 起始点所属的身体部分，0代表body candidate, 1代表face, 2代表hand
    end_point: 中止点
    """
    for i in range(len(smpl_joints_list)):
        joint3d = smpl_joints_list[i]
        anchor_x, anchor_y, anchor_z = joint3d[anchor_point]
        end_x, end_y, end_z = joint3d[end_point]
        
        vector_x = end_x - anchor_x
        vector_y = end_y - anchor_y
        vector_z = end_z - anchor_z
        offset_x = vector_x * alpha
        offset_y = vector_y * alpha
        offset_z = vector_z * alpha
        
        for affected_body_point in affected_body_points:
            joint3d[affected_body_point] = joint3d[affected_body_point] + torch.tensor([offset_x, offset_y, offset_z]).to(joint3d.device)


def pose_reshape_2d(alpha, pose, 
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
        if anchor_part == 0:
            anchor_x, anchor_y = candidate[anchor_point]
            if subset[anchor_point] == -1:
                continue
        elif anchor_part == 1:
            raise NotImplementedError("face anchor not implemented")
        elif anchor_part == 2:
            anchor_x, anchor_y = left_hand[anchor_point]
            if anchor_x <= 0 or anchor_y <= 0:
                continue
        elif anchor_part == 3:
            anchor_x, anchor_y = right_hand[anchor_point]
            if anchor_x <= 0 or anchor_y <= 0:
                continue

        if end_part == 0:
            end_x, end_y = candidate[end_point]
            if subset[end_point] == -1:
                continue
        elif end_part == 1:
            end_x, end_y = faces[end_point]
        elif end_part == 2:
            end_x, end_y = left_hand[end_point]
            if end_x <= 0 or end_y <= 0:
                continue
        elif end_part == 3:
            end_x, end_y = right_hand[end_point]
            if end_x <= 0 or end_y <= 0:
                continue
        
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

# reshapePool只负责形变，骨骼偏移、丢弃等得从draw层来做
class reshapePool3d:
    def __init__(self, reshape_type):  # 对每个视频只初始化一次
        self.reshape_type = reshape_type
        self.shoulder_alpha = 0
        self.upper_arm_alpha = 0
        self.forearm_alpha = 0
        self.body_alpha = 0
        self.thigh_alpha = 0
        self.calf_alpha = 0
        self.faces_indices = np.arange(0, 68)
        self.left_hands_indices = np.arange(0, 21)
        self.right_hands_indices = np.arange(0, 21)
        self.offset_2d_x = random.uniform(-1/30, 1/30)
        self.offset_2d_y = random.uniform(-1/120, 1/120)
        self.offset_3d_x = random.uniform(-150, 150)
        self.offset_3d_y = random.uniform(-150, 150)
        self.offset_3d_z = random.uniform(-250, 50)


        self.body_reshape_methods = [
            self.reshape_body,
            self.reshape_arm,
            self.reshape_leg,
            self.reshape_shoulder,
        ]
        
        options = ["normal_human", "baby", "long_arm", "long_leg", "random_long_arm_long_leg", "small_body"]
        if self.reshape_type == "low":
            weights = [0.75, 0,   0,   0.1, 0.1, 0.05]
            self.body_offset_selected_methods = []
        if self.reshape_type == "normal":
            weights = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
            self.body_offset_selected_methods = [self.offset_3d_all] if random.random() < 0.3 else []
        elif self.reshape_type == "high":
            weights = [0.375, 0.175, 0.1, 0.1, 0.125, 0.125]
            self.body_offset_selected_methods = [self.offset_3d_all] if random.random() < 0.4 else []
        choice = random.choices(options, weights=weights, k=1)[0]
        # print(f"debug: using body_type: {choice}")
        self.aug_init(choice)

        


    def aug_init(self, body_type):
        if body_type == "normal_human":
            self.body_reshape_selected_methods = []
        elif body_type == "baby":   # body不动
            self.upper_arm_alpha = -0.4
            self.forearm_alpha = -0.4
            self.shoulder_alpha = -0.3
            self.body_alpha = -0.2
            self.thigh_alpha = -0.4
            self.calf_alpha = -0.2
            self.body_reshape_selected_methods = [self.reshape_arm, self.reshape_leg, self.reshape_shoulder, self.reshape_body]
        elif body_type == "long_arm":
            self.upper_arm_alpha = 0.4
            self.forearm_alpha = 0.2
            self.shoulder_alpha = 0.3
            self.body_alpha = random.uniform(-0.5, 0)
            self.thigh_alpha = self.body_alpha / 2.5
            self.calf_alpha = self.body_alpha / 2.5
            self.body_reshape_selected_methods = [self.reshape_arm, self.reshape_leg, self.reshape_body]
        elif body_type == "long_leg":
            self.body_alpha = random.uniform(-0.2, 0.2)
            self.thigh_alpha = 0.4
            self.calf_alpha = 0.4
            self.shoulder_alpha = -0.2
            self.upper_arm_alpha = -0.2
            self.forearm_alpha = -0.2
            self.body_reshape_selected_methods = [self.reshape_leg, self.reshape_body, self.reshape_arm, self.reshape_shoulder]
        elif body_type == "small_body":
            self.body_alpha = -0.3
            self.thigh_alpha = -0.3
            self.calf_alpha = -0.3
            self.body_reshape_selected_methods = [self.reshape_body, self.reshape_leg]
        elif body_type == "random_long_arm_long_leg":
            self.upper_arm_alpha = random.uniform(-0.2, 0.6)
            self.forearm_alpha = random.uniform(-0.2, 0.6)
            self.thigh_alpha = random.uniform(-0.2, 0.6)
            self.calf_alpha = random.uniform(-0.2, 0.6)
            self.body_alpha = random.uniform(-0.6, 0.3)
            self.body_reshape_selected_methods = [self.reshape_body, self.reshape_arm, self.reshape_leg]


    def set_offset_3d_z(self, min_z):
        self.offset_3d_z = random.uniform(-min_z / 12, min_z / 12)



    def apply_random_reshapes(self, smpl_joints_list):
        # Apply the two selected reshape methods
        for method in self.body_reshape_selected_methods:
            method(smpl_joints_list)
        
        for method in self.body_offset_selected_methods:
            method(smpl_joints_list)



    def reshape_body(self, pose):
        pose_reshape_3d(self.body_alpha, pose,
                     12, 1, 
                     [1, 4, 7, 10]
                     )
        pose_reshape_3d(self.body_alpha, pose,
                    12, 2,
                    [2, 5, 8, 11]
                    )
        
    def reshape_arm(self, pose):
        pose_reshape_3d(self.upper_arm_alpha, pose,
                     16, 18,
                     [18, 20, 22]
                     )
        pose_reshape_3d(self.upper_arm_alpha, pose,
                    17, 19,
                    [19, 21, 23]
                    )
        pose_reshape_3d(self.forearm_alpha, pose,
                     18, 20,
                     [20, 22]
                     )
        pose_reshape_3d(self.forearm_alpha, pose,
                    19, 21,
                    [21, 23]
                    )


    def reshape_leg(self, pose):
        pose_reshape_3d(self.thigh_alpha, pose,
                     1, 4,
                     [4, 7, 10]
                     )
        pose_reshape_3d(self.thigh_alpha, pose,
                    2, 5,
                    [5, 8, 11]
                    )
        pose_reshape_3d(self.calf_alpha, pose,
                     4, 7,
                     [7, 10]
                     )
        pose_reshape_3d(self.calf_alpha, pose,
                    5, 8,
                    [8, 11]
                    )
        

    def reshape_shoulder(self, pose):
        pose_reshape_3d(self.shoulder_alpha, pose,
                     12, 16,
                     [16, 18, 20, 22]
                     )
        pose_reshape_3d(self.shoulder_alpha, pose,
                    12, 17,
                    [17, 19, 21, 23]
                    )
    
    def offset_3d_all(self, pose):
        pose = pose + torch.tensor([self.offset_3d_x, self.offset_3d_y, self.offset_3d_z]).to(pose.device)

    
        