"""
手势分类模块
封装 BiLSTM 模型加载和预测逻辑
"""

import numpy as np
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    """用于推理的 BiLSTM 分类网络定义。"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.2,
        bidirectional=True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_scale = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_scale, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_timestep = output[:, -1, :]
        logits = self.classifier(last_timestep)
        return logits


class GestureClassifier:
    """
    手势分类器
    负责加载训练好的 BiLSTM 模型并进行预测
    """

    def __init__(self, model_path, actions, sequence_length=30, confidence_threshold=0.90):
        """
        初始化手势分类器

        Args:
            model_path (str): 模型文件路径 (.pt 格式)
            actions (list): 动作名称列表
            sequence_length (int): 序列长度 (帧数)
            confidence_threshold (float): 置信度阈值 (0-1)
        """
        self.model_path = model_path
        self.actions = np.array(actions)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_torchscript = False
        self.model = None

        # 加载模型
        try:
            self.model = self._load_pytorch_model(model_path)
            self.model.eval()
            self.model.to(self.device)

            logger.info(f"成功加载 PyTorch 模型: {model_path}")
            logger.info(f"运行设备: {self.device}")
            logger.info(f"支持的手势: {', '.join(actions)}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _load_pytorch_model(self, model_path):
        """加载 PyTorch 模型，兼容 TorchScript 与 checkpoint。"""
        # 优先尝试加载 TorchScript，便于跨环境部署
        try:
            model = torch.jit.load(model_path, map_location=self.device)
            self.is_torchscript = True
            logger.info("检测到 TorchScript 模型")
            return model
        except Exception:
            self.is_torchscript = False

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, nn.Module):
            logger.info("检测到完整 nn.Module 模型")
            return checkpoint

        if not isinstance(checkpoint, dict):
            raise ValueError("不支持的 .pt 格式，请使用 TorchScript 或 checkpoint(dict)")

        model_state_dict = checkpoint.get('model_state_dict')
        if model_state_dict is None:
            raise ValueError("checkpoint 缺少 model_state_dict")

        input_dim = checkpoint.get('input_dim')
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_layers = checkpoint.get('num_layers', 2)
        num_classes = checkpoint.get('num_classes', len(self.actions))
        dropout = checkpoint.get('dropout', 0.2)
        bidirectional = checkpoint.get('bidirectional', True)

        if input_dim is None:
            raise ValueError("checkpoint 缺少 input_dim，无法重建 BiLSTM 结构")

        model = BiLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        model.load_state_dict(model_state_dict)
        logger.info("检测到 checkpoint(dict) 模型")
        return model

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
            with torch.no_grad():
                inputs = torch.tensor(sequence_array, dtype=torch.float32, device=self.device)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1)
                predictions = probs.squeeze(0).detach().cpu().numpy()
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
            "framework": "pytorch",
            "num_gestures": len(self.actions),
            "sequence_length": self.sequence_length,
            "confidence_threshold": self.confidence_threshold,
            "actions": self.actions.tolist(),
        }
