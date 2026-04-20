"""
MediaPipe 手部/身体检测模块
从原 code.ipynb 的 Cell 3 和 Cell 5 提取
"""

import cv2
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """
    使用 MediaPipe 进行手部和身体关键点检测
    """

    def __init__(self, min_detection_confidence=0.55, min_tracking_confidence=0.55):
        """
        初始化 MediaPipe 检测器

        Args:
            min_detection_confidence (float): 最小检测置信度
            min_tracking_confidence (float): 最小跟踪置信度
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # 创建 Holistic 对象
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image):
        """
        对输入图像进行手部和身体检测

        Args:
            image: OpenCV 格式的图像 (BGR)

        Returns:
            image: 处理后的图像
            results: MediaPipe 检测结果
        """
        # 颜色空间转换：BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # 进行检测
        results = self.holistic.process(image_rgb)

        # 恢复可写状态
        image_rgb.flags.writeable = True

        # 颜色空间转换：RGB -> BGR
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return image, results

    def draw_landmarks(self, image, results, use_styled=True):
        """
        在图像上绘制检测到的关键点

        Args:
            image: 输入图像
            results: MediaPipe 检测结果
            use_styled (bool): 是否使用风格化绘制
        """
        if use_styled:
            self._draw_styled_landmarks(image, results)
        else:
            self._draw_simple_landmarks(image, results)

    def _draw_simple_landmarks(self, image, results):
        """
        简单绘制关键点连接线 (原 code.ipynb Cell 4)
        """
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION
        )
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
        )
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
        )

    def _draw_styled_landmarks(self, image, results):
        """
        风格化绘制关键点 (原 code.ipynb Cell 5)
        带有自定义颜色和厚度
        """
        # 绘制面部关键点 (绿色)
        self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )

        # 绘制身体关键点 (红色)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

        # 绘制左手关键点 (紫色)
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )

        # 绘制右手关键点 (橙色)
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

    def release(self):
        """释放资源"""
        if self.holistic:
            self.holistic.close()

    def __del__(self):
        """析构函数，自动释放资源"""
        self.release()
