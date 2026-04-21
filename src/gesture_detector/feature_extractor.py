"""
关键点特征提取模块
单手版本：仅提取 pose + active hand（不含面部）
"""

import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    从 MediaPipe 检测结果中提取特征向量
    包含：姿态、单手关键点及几何特征
    """

    POSE_DIM = 33 * 4  # 132
    HAND_DIM = 21 * 3  # 63
    GEOMETRIC_FEATURES_DIM = 5  # 单手几何 + handedness flag

    def __init__(self):
        self.feature_dim = self.POSE_DIM + self.HAND_DIM + self.GEOMETRIC_FEATURES_DIM
        logger.info(f"特征向量维度(单手): {self.feature_dim}")

    def extract(self, results):
        keypoints = []

        pose = self._extract_pose(results)
        keypoints.extend(pose)

        hand = self._extract_active_hand(results)
        keypoints.extend(hand)

        geometric = self._extract_single_hand_geometric_features(results, hand)
        keypoints.extend(geometric)

        return np.array(keypoints)

    def _extract_pose(self, results):
        if results.pose_landmarks:
            return np.array(
                [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
            ).flatten()
        return np.zeros(self.POSE_DIM)

    def _extract_active_hand(self, results):
        active = getattr(results, "active_hand_landmarks", None)
        if active:
            return np.array(
                [[res.x, res.y, res.z] for res in active.landmark]
            ).flatten()

        # 兼容旧结构
        if getattr(results, "left_hand_landmarks", None):
            return np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
        if getattr(results, "right_hand_landmarks", None):
            return np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()

        return np.zeros(self.HAND_DIM)

    def _extract_single_hand_geometric_features(self, results, hand):
        """
        单手几何特征：
        - index-thumb distance
        - wrist-index distance
        - index方向与竖直方向夹角
        - 手掌中心深度均值
        - handedness flag (left=0, right=1, unknown=-1)
        """
        if np.allclose(hand, 0):
            return [0, 0, 0, 0, -1]

        try:
            hand_points = hand.reshape(-1, 3)
            wrist = hand_points[0]
            thumb_tip = hand_points[4]
            index_mcp = hand_points[5]
            index_tip = hand_points[8]
            palm_points = hand_points[[0, 5, 9, 13, 17]]

            index_thumb_dist = np.linalg.norm(index_tip - thumb_tip)
            wrist_index_dist = np.linalg.norm(index_tip - wrist)

            index_dir = index_tip - index_mcp
            up_vec = np.array([0, -1, 0])
            index_angle = (
                math.acos(np.dot(index_dir, up_vec) / (np.linalg.norm(index_dir) + 1e-6))
                if np.linalg.norm(index_dir) > 0
                else 0
            )

            palm_depth_mean = float(np.mean(palm_points[:, 2]))

            hand_label = getattr(results, "active_hand_label", "unknown")
            if hand_label == "left":
                handedness_flag = 0.0
            elif hand_label == "right":
                handedness_flag = 1.0
            else:
                handedness_flag = -1.0

            return [index_thumb_dist, wrist_index_dist, index_angle, palm_depth_mean, handedness_flag]
        except Exception as e:
            logger.warning(f"计算单手几何特征失败: {e}")
            return [0, 0, 0, 0, -1]

    def get_feature_dimension(self):
        return self.feature_dim
