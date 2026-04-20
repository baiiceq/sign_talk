"""
主程序入口
实时手势识别与交互控制系统

使用方法:
    python main.py [--config config.yaml]
"""

import cv2
import numpy as np
import argparse
import logging
import time
from collections import deque

from src.gesture_detector import MediaPipeDetector, FeatureExtractor, GestureClassifier
from src.control import CommandExecutor
from src.utils import setup_logger, ConfigLoader


class GestureControlSystem:
    """
    手势识别与交互控制系统
    整合检测、识别、控制的完整系统
    """

    def __init__(self, config_file='src/config/config.yaml'):
        """
        初始化系统

        Args:
            config_file (str): 配置文件路径
        """
        # 加载配置
        self.config = ConfigLoader(config_file)
        
        # 设置日志
        log_level = getattr(logging, self.config.get('logging.level', 'INFO'))
        log_file = self.config.get('logging.file')
        self.logger = setup_logger('GestureControlSystem', log_level, log_file)

        self.logger.info("=" * 50)
        self.logger.info("初始化手势识别与交互控制系统")
        self.logger.info("=" * 50)

        # 初始化检测器
        self.detector = MediaPipeDetector(
            min_detection_confidence=self.config.get('mediapipe.min_detection_confidence', 0.55),
            min_tracking_confidence=self.config.get('mediapipe.min_tracking_confidence', 0.55),
        )

        # 初始化特征提取器
        self.extractor = FeatureExtractor()

        # 初始化分类器
        model_path = self.config.get('model.path', 'models/gesture_model.keras')
        actions = self.config.get('model.actions', [])
        self.classifier = GestureClassifier(
            model_path=model_path,
            actions=actions,
            sequence_length=self.config.get('model.sequence_length', 30),
            confidence_threshold=self.config.get('model.confidence_threshold', 0.90),
        )

        # 初始化命令执行器
        self.executor = CommandExecutor()

        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.config.get('camera.device_id', 0))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('camera.frame_width', 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('camera.frame_height', 480))
        self.cap.set(cv2.CAP_PROP_FPS, self.config.get('camera.fps', 30))

        # 识别相关变量
        self.sequence = []
        self.gesture_history = deque(maxlen=self.config.get('display.max_history_count', 5))
        self.predictions = []
        self.previous_landmarks = None
        self.last_detected_time = 0
        self.frame_count = 0
        self.fps_time = time.time()

        self.logger.info("系统初始化完成")

    def process_frame(self, frame):
        """
        处理单帧图像

        Args:
            frame: 输入图像

        Returns:
            frame: 处理后的图像
            gesture: 检测到的手势 (如果有)
        """
        # 手部和身体检测
        image, results = self.detector.detect(frame)

        # 绘制关键点
        if self.config.get('mediapipe.draw_landmarks', True):
            self.detector.draw_landmarks(
                image, results, use_styled=self.config.get('mediapipe.use_styled_drawing', True)
            )

        # 提取特征
        current_landmarks = self.extractor.extract(results)

        # 动作检测：如果动作变化不足，跳过处理
        motion_threshold = self.config.get('recognition.motion_threshold', 0.01)
        if self.previous_landmarks is not None:
            landmark_diff = np.abs(current_landmarks - self.previous_landmarks).sum()
            if landmark_diff < motion_threshold:
                self.previous_landmarks = current_landmarks
                return image, None

        self.previous_landmarks = current_landmarks

        # 检查暂停期
        current_time = time.time()
        pause_time = self.config.get('recognition.pause_time', 2.0)
        if current_time - self.last_detected_time < pause_time:
            return image, None

        # 构建序列
        self.sequence.append(current_landmarks)
        max_seq_len = self.config.get('recognition.max_sequence_length', 30)
        if len(self.sequence) > max_seq_len:
            self.sequence = self.sequence[-max_seq_len:]

        # 执行预测
        min_seq_len = self.config.get('recognition.min_sequence_length', 10)
        gesture = None

        if len(self.sequence) >= min_seq_len:
            gesture_pred, confidence, _ = self.classifier.predict(self.sequence)

            if gesture_pred:
                self.predictions.append(gesture_pred)

                # 检查是否识别完整手势
                if len(self.predictions) >= min_seq_len:
                    most_common = max(
                        set(self.predictions[-min_seq_len:]),
                        key=self.predictions[-min_seq_len:].count,
                    )
                    count = self.predictions[-min_seq_len:].count(most_common)
                    threshold_count = int(min_seq_len * self.config.get('model.confidence_threshold', 0.90))

                    if count >= threshold_count:
                        gesture = most_common
                        self.gesture_history.append(gesture)
                        self.sequence = []
                        self.predictions = []
                        self.last_detected_time = current_time
                        self.logger.info(f"识别到手势: {gesture}")

        return image, gesture

    def draw_info(self, frame, gesture=None):
        """
        在画面上绘制信息

        Args:
            frame: 输入图像
            gesture: 当前检测到的手势

        Returns:
            frame: 绘制后的图像
        """
        # 计算 FPS
        if self.frame_count % 10 == 0:
            elapsed = time.time() - self.fps_time
            self.fps = 10 / elapsed if elapsed > 0 else 0
            self.fps_time = time.time()

        self.frame_count += 1

        # 绘制 FPS
        if self.config.get('display.show_fps', True):
            cv2.putText(
                frame,
                f'FPS: {self.fps:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # 绘制当前手势
        if gesture and self.config.get('display.show_confidence', True):
            cv2.putText(
                frame,
                f'Gesture: {gesture}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

        # 绘制手势历史
        if self.config.get('display.show_gesture_history', True) and self.gesture_history:
            history_text = ' -> '.join(list(self.gesture_history))
            cv2.putText(
                frame,
                f'History: {history_text}',
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

        return frame

    def run(self):
        """运行系统"""
        self.logger.info("系统启动，按 'q' 键退出")

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                if not ret:
                    self.logger.error("读取摄像头失败")
                    break

                # 处理图像
                image, gesture = self.process_frame(frame)

                # 绘制信息
                image = self.draw_info(image, gesture)

                # 执行命令
                if gesture and self.config.get('control.enable_mouse', True):
                    self.executor.execute(gesture)

                # 显示画面
                if self.config.get('display.show_frame', True):
                    cv2.imshow('Gesture Recognition Control System', image)

                # 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("用户退出")
                    break

        except KeyboardInterrupt:
            self.logger.info("接收到中断信号")
        except Exception as e:
            self.logger.error(f"系统错误: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        self.logger.info("清理资源...")
        if self.cap:
            self.cap.release()
        if self.detector:
            self.detector.release()
        cv2.destroyAllWindows()
        self.logger.info("系统已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手势识别与交互控制系统')
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/config.yaml',
        help='配置文件路径',
    )
    args = parser.parse_args()

    # 创建并运行系统
    system = GestureControlSystem(config_file=args.config)
    system.run()


if __name__ == '__main__':
    main()
