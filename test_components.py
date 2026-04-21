"""
快速开发和测试脚本
用于验证系统各个模块的功能
"""

import sys
import logging
from src.utils import setup_logger, ConfigLoader
from src.gesture_detector import MediaPipeDetector, FeatureExtractor, GestureClassifier
from src.control import MouseController, KeyboardController, CommandExecutor

# 设置日志
logger = setup_logger('test_script', logging.INFO)


def test_config_loader():
    """测试配置加载"""
    logger.info("测试配置加载...")
    config = ConfigLoader('src/config/config.yaml')
    logger.info(f"模型路径: {config.get('model.path')}")
    logger.info(f"摄像头设备: {config.get('camera.device_id')}")
    logger.info("✓ 配置加载成功\n")


def test_mediapipe_detector():
    """测试 MediaPipe 检测器"""
    logger.info("测试 MediaPipe 检测器...")
    try:
        detector = MediaPipeDetector()
        logger.info("✓ 检测器初始化成功")
        detector.release()
        logger.info("✓ 检测器释放成功\n")
    except Exception as e:
        logger.error(f"✗ 检测器测试失败: {e}\n")


def test_feature_extractor():
    """测试特征提取器"""
    logger.info("测试特征提取器...")
    try:
        extractor = FeatureExtractor()
        dim = extractor.get_feature_dimension()
        logger.info(f"✓ 特征向量维度: {dim}")
        logger.info("✓ 特征提取器初始化成功\n")
    except Exception as e:
        logger.error(f"✗ 特征提取器测试失败: {e}\n")


def test_gesture_classifier():
    """测试手势分类器"""
    logger.info("测试手势分类器...")
    try:
        # 注意：实际运行时需要模型文件存在
        classifier = GestureClassifier(
            model_path='models/gesture_model.pt',
            actions=['gesture1', 'gesture2'],
            sequence_length=30,
        )
        logger.info("✓ 分类器初始化成功")
        logger.info(f"  支持手势: {classifier.get_actions()}")
        logger.info("✓ 手势分类器测试通过\n")
    except FileNotFoundError:
        logger.warning("⚠ 模型文件不存在，跳过分类器测试\n")
    except Exception as e:
        logger.error(f"✗ 分类器测试失败: {e}\n")


def test_mouse_controller():
    """测试鼠标控制器"""
    logger.info("测试鼠标控制器...")
    try:
        mouse = MouseController(speed=1.0)
        pos = mouse.get_position()
        logger.info(f"✓ 鼠标当前位置: {pos}")
        logger.info("✓ 鼠标控制器初始化成功\n")
    except Exception as e:
        logger.error(f"✗ 鼠标控制器测试失败: {e}\n")


def test_keyboard_controller():
    """测试键盘控制器"""
    logger.info("测试键盘控制器...")
    try:
        keyboard = KeyboardController()
        logger.info("✓ 键盘控制器初始化成功")
        logger.info(f"  支持的按键: {list(keyboard.KEY_MAP.keys())[:10]}...")
        logger.info("✓ 键盘控制器测试通过\n")
    except Exception as e:
        logger.error(f"✗ 键盘控制器测试失败: {e}\n")


def test_command_executor():
    """测试命令执行器"""
    logger.info("测试命令执行器...")
    try:
        executor = CommandExecutor()
        gestures = executor.get_supported_gestures()
        logger.info(f"✓ 支持的手势: {gestures[:5]}...")
        logger.info(f"✓ 总共支持 {len(gestures)} 个命令")
        logger.info("✓ 命令执行器测试通过\n")
    except Exception as e:
        logger.error(f"✗ 命令执行器测试失败: {e}\n")


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("开始系统组件测试")
    logger.info("=" * 60 + "\n")

    # 运行各项测试
    test_config_loader()
    test_mediapipe_detector()
    test_feature_extractor()
    test_gesture_classifier()
    test_mouse_controller()
    test_keyboard_controller()
    test_command_executor()

    logger.info("=" * 60)
    logger.info("系统组件测试完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
