"""
手势分类模块
封装 BiLSTM 模型加载和预测逻辑
"""

import numpy as np
from tensorflow.keras.models import load_model
import logging

logger = logging.getLogger(__name__)


class GestureClassifier:
    """
    手势分类器
    负责加载训练好的 BiLSTM 模型并进行预测
    """

    def __init__(self, model_path, actions, sequence_length=30, confidence_threshold=0.90):
        """
        初始化手势分类器

        Args:
            model_path (str): 模型文件路径 (.keras 格式)
            actions (list): 动作名称列表
            sequence_length (int): 序列长度 (帧数)
            confidence_threshold (float): 置信度阈值 (0-1)
        """
        self.model_path = model_path
        self.actions = np.array(actions)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold

        # 加载模型
        try:
            self.model = load_model(model_path)
            logger.info(f"成功加载模型: {model_path}")
            logger.info(f"支持的手势: {', '.join(actions)}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def predict(self, sequence):
        """
        对一个序列进行手势预测

        Args:
            sequence (list or ndarray): 特征向量序列，形状为 (sequence_length, feature_dim)

        Returns:
            gesture (str): 预测的手势名称
            confidence (float): 预测置信度
            all_probs (ndarray): 所有类别的预测概率
        """
        if len(sequence) != self.sequence_length:
            logger.warning(f"序列长度不匹配: 期望 {self.sequence_length}, 得到 {len(sequence)}")
            return None, 0.0, None

        # 准备输入 (添加 batch 维度)
        sequence_array = np.expand_dims(np.array(sequence), axis=0)

        # 预测
        try:
            predictions = self.model.predict(sequence_array, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]

            gesture = self.actions[predicted_idx]

            return gesture, confidence, predictions
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None, 0.0, None

    def predict_batch(self, sequences):
        """
        批量预测多个序列

        Args:
            sequences (list): 序列列表

        Returns:
            results (list): 预测结果列表，每个元素为 (gesture, confidence)
        """
        results = []
        for sequence in sequences:
            gesture, confidence, _ = self.predict(sequence)
            results.append((gesture, confidence))
        return results

    def get_top_n_predictions(self, sequence, n=3):
        """
        获取置信度最高的 N 个预测

        Args:
            sequence (list): 特征向量序列
            n (int): 返回前 N 个预测

        Returns:
            top_predictions (list): 前N个预测，每个元素为 (gesture, confidence)
        """
        gesture, confidence, all_probs = self.predict(sequence)

        if all_probs is None:
            return []

        # 获取最高的 N 个索引
        top_n_indices = np.argsort(all_probs)[-n:][::-1]

        top_predictions = [
            (self.actions[idx], all_probs[idx]) for idx in top_n_indices
        ]

        return top_predictions

    def filter_by_confidence(self, gesture, confidence):
        """
        根据置信度阈值过滤预测结果

        Args:
            gesture (str): 手势名称
            confidence (float): 置信度

        Returns:
            bool: 是否超过阈值
        """
        return confidence >= self.confidence_threshold

    def get_actions(self):
        """获取支持的所有动作"""
        return self.actions.tolist()

    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "num_gestures": len(self.actions),
            "sequence_length": self.sequence_length,
            "confidence_threshold": self.confidence_threshold,
            "actions": self.actions.tolist(),
        }
