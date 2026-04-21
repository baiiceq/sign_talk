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
        model_path = self.config.get('model.path', 'models/gesture_model.pt')
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
        
        # 手势检测变量
        self.gesture_active = False
        self.gesture_frames = []
        self.static_gesture_counter = 0
        self.last_prediction = None

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

        # 手势开始/结束检测
        motion_threshold = self.config.get('recognition.motion_threshold', 0.01)
        gesture_start_threshold = self.config.get('recognition.gesture_start_threshold', 0.05)
        gesture_end_threshold = self.config.get('recognition.gesture_end_threshold', 0.02)
        min_gesture_frames = self.config.get('recognition.min_gesture_frames', 10)
        static_gesture_timeout = self.config.get('recognition.static_gesture_timeout', 60)  # 静态手势超时帧数

        # 检测是否有手存在
        has_hand = results.active_hand_landmarks is not None
        
        if has_hand:
            # 如果之前没有手，开始新序列
            if not self.gesture_active:
                self.gesture_active = True
                self.gesture_frames = []
                self.static_gesture_counter = 0
                self.last_prediction = None
                self.logger.info("检测到手，开始手势收集")
            
            # 收集当前帧
            self.gesture_frames.append(current_landmarks)
            
            # 保持最大长度
            max_len = self.config.get('recognition.max_sequence_length', 30)
            if len(self.gesture_frames) > max_len:
                self.gesture_frames = self.gesture_frames[-max_len:]
            
            # 当序列足够长时，进行实时预测
            if len(self.gesture_frames) >= min_gesture_frames:
                gesture_pred, confidence, _ = self.classifier.predict(np.array(self.gesture_frames))
                
                if gesture_pred and confidence > self.config.get('model.confidence_threshold', 0.90):
                    # 如果预测结果稳定，计数器增加
                    if gesture_pred == self.last_prediction:
                        self.static_gesture_counter += 1
                    else:
                        self.static_gesture_counter = 1
                        self.last_prediction = gesture_pred
                    
                    # 如果连续预测到相同手势足够次数，触发
                    stable_threshold = self.config.get('recognition.stable_prediction_count', 5)
                    if self.static_gesture_counter >= stable_threshold:
                        self.gesture_history.append(gesture_pred)
                        self.logger.info(f"识别到静态手势: {gesture_pred}")
                        self.gesture_active = False
                        self.gesture_frames = []
                        self.static_gesture_counter = 0
                        self.last_prediction = None
                        return image, gesture_pred
                else:
                    # 重置计数器如果预测不稳定
                    self.static_gesture_counter = 0
                    self.last_prediction = None
            
            # 超时检查：如果太久没有识别到手势，重置
            if len(self.gesture_frames) > static_gesture_timeout:
                self.gesture_active = False
                self.gesture_frames = []
                self.static_gesture_counter = 0
                self.last_prediction = None
        else:
            # 没有手时，重置状态
            if self.gesture_active:
                self.gesture_active = False
                self.gesture_frames = []
                self.static_gesture_counter = 0
                self.last_prediction = None

        self.previous_landmarks = current_landmarks
        return image, None

    def process_gesture_sequence(self, gesture_frames):
        """
        处理检测到的手势序列

        Args:
            gesture_frames: 手势帧列表

        Returns:
            gesture: 识别到的手势或None
        """
        if len(gesture_frames) < 10:
            return None
            
        # 填充到固定长度
        sequence = gesture_frames[:30]  # 取前30帧
        if len(sequence) < 30:
            # 重复最后一帧填充
            last_frame = sequence[-1]
            while len(sequence) < 30:
                sequence.append(last_frame)
        
        sequence_array = np.array(sequence)
        
        # 使用分类器预测
        gesture_pred, confidence, _ = self.classifier.predict(sequence_array)
        
        self.previous_landmarks = current_landmarks
        return image, None

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
