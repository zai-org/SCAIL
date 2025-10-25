import numpy as np
import random
from NLFPoseExtract.nlf_draw import intrinsic_matrix_from_field_of_view, process_data_to_COCO_format, p3d_to_p2d
import torch



# reshapePool只负责形变，骨骼偏移、丢弃等得从draw层来做
class reshapePool3d:
    def __init__(self, reshape_type, height, width):  # 对每个视频只初始化一次
        self.reshape_type = reshape_type
        self.height = height
        self.width = width
        self.shoulder_alpha = 0
        self.upper_arm_alpha = 0
        self.forearm_alpha = 0
        self.body_alpha = 0
        self.thigh_alpha = 0
        self.calf_alpha = 0
        self.face_alpha = random.choices([-0.4, -0.2, 0, 0.2, 0.4], weights=[0.2, 0.15, 0.3, 0.15, 0.2], k=1)[0]


        self.body_reshape_methods = [
            self.reshape_body,
            self.reshape_arm,
            self.reshape_leg,
            self.reshape_shoulder,
        ]

        self.face_reshape_methods = [
            self.reshape_face,
        ]

        self.body_offset_selected_methods = []
        
        options = ["normal_human", "dwarf", "slender", "elf", "random_long_arm_long_leg", "king-kong"]
        if self.reshape_type == "low":
            weights = [0.8, 0,   0,   0.1, 0.1, 0]
            self.body_offset_selected_methods = []
        if self.reshape_type == "normal":
            weights = [0.4, 0.1, 0.1, 0.1, 0.2, 0.1]
        elif self.reshape_type == "high":
            weights = [0.3, 0.2, 0.1, 0.1, 0.2, 0.1]
        choice = random.choices(options, weights=weights, k=1)[0]
        self.aug_init(choice)


    def pose_reshape_2d_for_face(self, alpha, candidate, face, subset, body_anchor_point, body_affected_points):
        anchor_x, anchor_y = candidate[body_anchor_point]
        if subset[body_anchor_point] == -1:
            return

        for face_point_idx in range(len(face)):
            face_point_x, face_point_y = face[face_point_idx]
            if face_point_x == -1 or face_point_y == -1:
                continue
            vector_x = face_point_x - anchor_x
            vector_y = face_point_y - anchor_y
            offset_x = vector_x * alpha
            offset_y = vector_y * alpha
            face[face_point_idx] = [face_point_x + offset_x, face_point_y + offset_y]

        for body_affected_point_idx in body_affected_points:
            body_point_x, body_point_y = candidate[body_affected_point_idx]
            if subset[body_affected_point_idx] == -1 or body_point_x == -1 or body_point_y == -1:
                continue
            vector_x = body_point_x - anchor_x
            vector_y = body_point_y - anchor_y
            offset_x = vector_x * alpha
            offset_y = vector_y * alpha
            candidate[body_affected_point_idx] = [body_point_x + offset_x, body_point_y + offset_y]

    def pose_reshape_3d(self, alpha, smpl_joints, candidate, subset, left_hand, right_hand, face,
                    anchor_point, end_point, 
                    affected_body_points):
        """
        # joints3d: n, 24, 3
        alpha: 变化比例
        anchor_point: 起始点
        anchor_part: 起始点所属的身体部分，0代表body candidate, 1代表face, 2代表hand
        end_point: 中止点
        """
        if torch.sum(smpl_joints[anchor_point]) == 0:
            right_hand[:] = -1
            left_hand[:] = -1
            face[:] = -1
            subset[:] = -1
            return


        anchor_x, anchor_y, anchor_z = smpl_joints[anchor_point]
        end_x, end_y, end_z = smpl_joints[end_point]
        
        vector_x = end_x - anchor_x
        vector_y = end_y - anchor_y
        vector_z = end_z - anchor_z
        offset_x = (vector_x * alpha).item()
        offset_y = (vector_y * alpha).item()
        offset_z = (vector_z * alpha).item()

        map_to_2d = {}
        map_to_2d[4] = 11
        map_to_2d[7] = 12
        map_to_2d[10] = 13
        map_to_2d[5] = 8
        map_to_2d[8] = 9
        map_to_2d[11] = 10
        map_to_2d[20] = 6
        map_to_2d[22] = 7
        map_to_2d[21] = 3
        map_to_2d[23] = 4
        map_to_2d[18] = 5
        map_to_2d[19] = 2
        

        for affected_body_point in affected_body_points:
            if torch.sum(smpl_joints[affected_body_point]) == 0:
                continue
            new_smpl_joint = smpl_joints[affected_body_point] + torch.tensor([offset_x, offset_y, offset_z]).to(smpl_joints.device)
            new_smpl_joint_2d_offset = p3d_to_p2d(new_smpl_joint.reshape(1,1,3).cpu().numpy(), self.height, self.width)[0][0] - p3d_to_p2d(smpl_joints[affected_body_point].reshape(1,1,3).cpu().numpy(), self.height, self.width)[0][0]
            new_smpl_joint_2d_offset = np.array([new_smpl_joint_2d_offset[0] / self.width, new_smpl_joint_2d_offset[1] / self.height])
            smpl_joints[affected_body_point] = new_smpl_joint
            if affected_body_point in map_to_2d.keys():
                affected_candidate_point_idx = map_to_2d[affected_body_point]
                if subset[affected_candidate_point_idx] != -1 and candidate[affected_candidate_point_idx][0] != -1 and candidate[affected_candidate_point_idx][1] != -1:
                    candidate[affected_candidate_point_idx] = candidate[affected_candidate_point_idx] + new_smpl_joint_2d_offset           # 2d的也移动这么多
                if affected_candidate_point_idx == 4:   # dwpose 右手 （反的
                    left_hand[:] = left_hand + new_smpl_joint_2d_offset
                if affected_candidate_point_idx == 7:   # dwpose 左手 （反的
                    right_hand[:] = right_hand + new_smpl_joint_2d_offset



    def aug_init(self, body_type):
        print(f"augmentation: using body_type: {body_type}")
        self.shoulder_alpha = 0
        self.upper_arm_alpha = 0
        self.forearm_alpha = 0
        self.body_alpha = 0
        self.thigh_alpha = 0
        self.calf_alpha = 0
        if body_type == "normal_human":
            self.body_reshape_selected_methods = []
        elif body_type == "dwarf":   # body不动
            self.upper_arm_alpha = random.uniform(-0.3, -0.2)
            self.forearm_alpha = self.upper_arm_alpha
            self.shoulder_alpha = -0.2
            self.thigh_alpha = random.uniform(-0.3, -0.2)
            self.calf_alpha = self.thigh_alpha
            self.body_reshape_selected_methods = [self.reshape_shoulder, self.reshape_arm, self.reshape_leg]
            self.face_alpha = 0.2
        elif body_type == "slender":
            self.upper_arm_alpha = 0.3
            self.forearm_alpha = 0.2
            self.shoulder_alpha = 0.2
            self.thigh_alpha = 0.1
            self.calf_alpha = 0.1
            self.body_reshape_selected_methods = [self.reshape_shoulder, self.reshape_arm, self.reshape_leg]
            self.face_alpha = -0.2
        elif body_type == "elf":
            self.body_alpha = random.uniform(-0.2, 0.2)
            self.shoulder_alpha = 0.1
            self.upper_arm_alpha = 0.1
            self.forearm_alpha = 0.1
            self.thigh_alpha = 0.25
            self.calf_alpha = 0.25
            self.body_reshape_selected_methods = [self.reshape_body, self.reshape_shoulder, self.reshape_arm, self.reshape_leg]
            self.face_alpha = 0
        elif body_type == "king-kong":
            self.body_alpha = 0.1
            self.thigh_alpha = -0.25
            self.calf_alpha = -0.25
            self.upper_arm_alpha = 0.2
            self.forearm_alpha = 0.2
            self.shoulder_alpha = 0.3
            self.body_reshape_selected_methods = [self.reshape_body, self.reshape_shoulder, self.reshape_arm, self.reshape_leg]
            self.face_alpha = 0
        elif body_type == "random_long_arm_long_leg":
            self.upper_arm_alpha = random.uniform(-0.2, 0.2)
            self.forearm_alpha = random.uniform(-0.2, 0.2)
            self.thigh_alpha = random.uniform(-0.2, 0.2)
            self.calf_alpha = random.uniform(-0.2, 0.2)
            self.body_alpha = random.uniform(-0.1, 0.1)
            self.body_reshape_selected_methods = [self.reshape_body, self.reshape_arm, self.reshape_leg]
        # elif body_type == "test_case":
        #     self.upper_arm_alpha = -0.4
        #     self.forearm_alpha = -0.4
        #     self.shoulder_alpha = -0.3
        #     self.thigh_alpha = 0.2
        #     self.calf_alpha = 0.2
        #     self.body_alpha = 0.1
        #     self.body_reshape_selected_methods = [self.reshape_body, self.reshape_shoulder, self.reshape_arm, self.reshape_leg]



    def apply_random_reshapes(self, smpl_joints_list, candidate, left_hand, right_hand, face, subset):
        # Apply the two selected reshape methods
        for method in self.body_reshape_selected_methods:
            method(smpl_joints_list, candidate, subset, left_hand, right_hand, face)

        for method in self.body_offset_selected_methods:
            method(smpl_joints_list, candidate, left_hand, right_hand, face)

        for method in self.face_reshape_methods:
            method(candidate, face, subset)
        



    def reshape_body(self, smpl_joints_list, candidate, subset, left_hand, right_hand, face):
        self.pose_reshape_3d(self.body_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                     12, 1,
                     [1, 4, 7, 10]
                     )
        self.pose_reshape_3d(self.body_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                    12, 2,
                    [2, 5, 8, 11]
                    )

    def reshape_arm(self, smpl_joints_list, candidate, subset, left_hand, right_hand, face):
        self.pose_reshape_3d(self.upper_arm_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                     16, 18,
                     [18, 20, 22]
                     )
        self.pose_reshape_3d(self.upper_arm_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                    17, 19,
                    [19, 21, 23]
                    )
        self.pose_reshape_3d(self.forearm_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                     18, 20,
                     [20, 22]
                     )
        self.pose_reshape_3d(self.forearm_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                    19, 21,
                    [21, 23]
                    )


    def reshape_leg(self, smpl_joints_list, candidate, subset, left_hand, right_hand, face):
        self.pose_reshape_3d(self.thigh_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                     1, 4,
                     [4, 7, 10]
                     )
        self.pose_reshape_3d(self.thigh_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                    2, 5,
                    [5, 8, 11]
                    )
        self.pose_reshape_3d(self.calf_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                     4, 7,
                     [7, 10]
                     )
        self.pose_reshape_3d(self.calf_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                    5, 8,
                    [8, 11]
                    )
        

    def reshape_shoulder(self, smpl_joints_list, candidate, subset, left_hand, right_hand, face):
        self.pose_reshape_3d(self.shoulder_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                     12, 16,
                     [16, 18, 20, 22]
                     )
        self.pose_reshape_3d(self.shoulder_alpha, smpl_joints_list, candidate, subset, left_hand, right_hand, face,
                    12, 17,
                    [17, 19, 21, 23]
                    )
    
    # def offset_3d_all(self, smpl_joints_list, candidate, left_hand, right_hand, face):
    #     smpl_joints_list = smpl_joints_list + torch.tensor([self.offset_3d_x, self.offset_3d_y, self.offset_3d_z]).to(smpl_joints_list.device)
    #     offset_2d = p3d_to_p2d(np.array([[[self.offset_3d_x, self.offset_3d_y, self.offset_3d_z]]]), self.height, self.width)[0][0]
    #     candidate = candidate + np.array([offset_2d[0], offset_2d[1]])
    #     left_hand = left_hand + np.array([offset_2d[0], offset_2d[1]])
    #     right_hand = right_hand + np.array([offset_2d[0], offset_2d[1]])
    #     face = face + np.array([offset_2d[0], offset_2d[1]])


    def reshape_face(self, candidate, face, subset):
        self.pose_reshape_2d_for_face(alpha=self.face_alpha, candidate=candidate, face=face, subset=subset, 
                                      body_anchor_point=0, body_affected_points=[14, 15, 16, 17])
