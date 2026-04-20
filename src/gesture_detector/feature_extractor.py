"""
关键点特征提取模块
从原 code.ipynb 的 Cell 13 提取并增强
"""

import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    从 MediaPipe 检测结果中提取特征向量
    包含：姿态、面部、手部关键点及几何特征
    """

    # 关键点维度常数
    POSE_DIM = 33 * 4  # 33个姿态点 × (x, y, z, visibility)
    FACE_DIM = 468 * 3  # 468个面部点 × (x, y, z)
    HAND_DIM = 21 * 3  # 21个手部点 × (x, y, z)
    GEOMETRIC_FEATURES_DIM = 6  # 几何特征数量

    def __init__(self):
        """初始化特征提取器"""
        self.feature_dim = (
            self.POSE_DIM + self.FACE_DIM + self.HAND_DIM * 2 + self.GEOMETRIC_FEATURES_DIM
        )
        logger.info(f"特征向量维度: {self.feature_dim}")

    def extract(self, results):
        """
        从检测结果中提取完整特征向量

        Args:
            results: MediaPipe 检测结果

        Returns:
            keypoints: 特征向量 (numpy array)
        """
        keypoints = []

        # 1. 提取姿态关键点 (33个点 × 4维 = 132维)
        pose = self._extract_pose(results)
        keypoints.extend(pose)

        # 2. 提取面部关键点 (468个点 × 3维 = 1404维)
        face = self._extract_face(results)
        keypoints.extend(face)

        # 3. 提取左手关键点 (21个点 × 3维 = 63维)
        lh = self._extract_left_hand(results)
        keypoints.extend(lh)

        # 4. 提取右手关键点 (21个点 × 3维 = 63维)
        rh = self._extract_right_hand(results)
        keypoints.extend(rh)

        # 5. 计算几何特征 (手-脸距离、手-手距离等)
        geometric_features = self._extract_geometric_features(results, face, lh, rh)
        keypoints.extend(geometric_features)

        return np.array(keypoints)

    def _extract_pose(self, results):
        """提取姿态关键点"""
        if results.pose_landmarks:
            return np.array(
                [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
            ).flatten()
        else:
            return np.zeros(self.POSE_DIM)

    def _extract_face(self, results):
        """提取面部关键点"""
        if results.face_landmarks:
            return np.array(
                [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
            ).flatten()
        else:
            return np.zeros(self.FACE_DIM)

    def _extract_left_hand(self, results):
        """提取左手关键点"""
        if results.left_hand_landmarks:
            return np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
        else:
            return np.zeros(self.HAND_DIM)

    def _extract_right_hand(self, results):
        """提取右手关键点"""
        if results.right_hand_landmarks:
            return np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
        else:
            return np.zeros(self.HAND_DIM)

    def _extract_geometric_features(self, results, face, lh, rh):
        """
        提取几何特征：
        - 左手到脸部距离
        - 右手到脸部距离
        - 左右手之间距离
        - 手指尖之间距离
        - 左手指尖到脸部角度
        - 右手指尖到脸部角度
        """
        features = []

        if results.face_landmarks and results.left_hand_landmarks and results.right_hand_landmarks:
            # 计算各部分中心点
            face_center = np.mean(face.reshape(-1, 3), axis=0)
            lh_center = np.mean(lh.reshape(-1, 3), axis=0)
            rh_center = np.mean(rh.reshape(-1, 3), axis=0)

            # 距离特征
            lh_face_dist = np.linalg.norm(lh_center - face_center)
            rh_face_dist = np.linalg.norm(rh_center - face_center)
            lh_rh_dist = np.linalg.norm(lh_center - rh_center)
            features.extend([lh_face_dist, rh_face_dist, lh_rh_dist])

            # 手指尖特征 (索引指 = 第16个点)
            try:
                lh_finger_tip = lh[16 * 3 : 17 * 3]
                rh_finger_tip = rh[16 * 3 : 17 * 3]
                finger_tip_diff = np.linalg.norm(lh_finger_tip - rh_finger_tip)
                features.append(finger_tip_diff)

                # 手指到脸部的角度特征
                lh_to_face = face_center - lh_finger_tip
                rh_to_face = face_center - rh_finger_tip

                lh_angle = (
                    math.acos(
                        np.dot(lh_to_face, [0, 1, 0])
                        / (np.linalg.norm(lh_to_face) * 1 + 1e-6)
                    )
                    if np.linalg.norm(lh_to_face) > 0
                    else 0
                )
                rh_angle = (
                    math.acos(
                        np.dot(rh_to_face, [0, 1, 0])
                        / (np.linalg.norm(rh_to_face) * 1 + 1e-6)
                    )
                    if np.linalg.norm(rh_to_face) > 0
                    else 0
                )
                features.extend([lh_angle, rh_angle])
            except Exception as e:
                logger.warning(f"计算角度特征失败: {e}")
                features.extend([0, 0])
        else:
            # 如果缺少关键点，填充零值
            features = [0] * self.GEOMETRIC_FEATURES_DIM

        return features

    def get_feature_dimension(self):
        """获取特征向量的维度"""
        return self.feature_dim
