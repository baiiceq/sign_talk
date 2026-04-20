"""
鼠标控制模块
通过手势控制系统鼠标
"""

import pyautogui
import logging

logger = logging.getLogger(__name__)

# 禁用 PyAutoGUI 的故障保全 (移动到屏幕角落时不停止)
pyautogui.FAIL_SAFE = False


class MouseController:
    """
    鼠标控制器
    支持移动、点击、滚动等操作
    """

    def __init__(self, speed=1.0):
        """
        初始化鼠标控制器

        Args:
            speed (float): 鼠标移动速度倍数 (1.0 为正常速度)
        """
        self.speed = speed
        logger.info(f"鼠标控制器已初始化 (速度倍数: {speed})")

    def move(self, dx, dy):
        """
        相对移动鼠标

        Args:
            dx (int): 水平移动距离 (像素)
            dy (int): 竖直移动距离 (像素)
        """
        try:
            pyautogui.move(int(dx * self.speed), int(dy * self.speed), duration=0.1)
        except Exception as e:
            logger.error(f"鼠标移动失败: {e}")

    def move_to(self, x, y):
        """
        移动鼠标到绝对位置

        Args:
            x (int): 目标X坐标
            y (int): 目标Y坐标
        """
        try:
            pyautogui.moveTo(int(x), int(y), duration=0.1)
        except Exception as e:
            logger.error(f"鼠标定位失败: {e}")

    def click(self, button='left', clicks=1):
        """
        点击鼠标

        Args:
            button (str): 鼠标按钮 ('left', 'right', 'middle')
            clicks (int): 点击次数 (1=单击, 2=双击)
        """
        try:
            pyautogui.click(button=button, clicks=clicks)
            logger.debug(f"鼠标{button}键点击: {clicks}次")
        except Exception as e:
            logger.error(f"鼠标点击失败: {e}")

    def left_click(self, clicks=1):
        """左键点击"""
        self.click(button='left', clicks=clicks)

    def right_click(self):
        """右键点击"""
        self.click(button='right', clicks=1)

    def double_click(self):
        """双击"""
        self.click(button='left', clicks=2)

    def scroll(self, dy, dx=0):
        """
        滚动鼠标滚轮

        Args:
            dy (int): 竖直滚动距离 (正数向上, 负数向下)
            dx (int): 水平滚动距离
        """
        try:
            pyautogui.scroll(dy)
            if dx != 0:
                pyautogui.scroll(dx, x=0)
            logger.debug(f"鼠标滚动: dy={dy}, dx={dx}")
        except Exception as e:
            logger.error(f"鼠标滚动失败: {e}")

    def scroll_up(self, amount=3):
        """向上滚动"""
        self.scroll(amount)

    def scroll_down(self, amount=3):
        """向下滚动"""
        self.scroll(-amount)

    def drag(self, dx, dy, duration=0.5):
        """
        拖拽鼠标

        Args:
            dx (int): 水平拖拽距离
            dy (int): 竖直拖拽距离
            duration (float): 拖拽持续时间 (秒)
        """
        try:
            pyautogui.drag(int(dx * self.speed), int(dy * self.speed), duration=duration)
            logger.debug(f"鼠标拖拽: dx={dx}, dy={dy}")
        except Exception as e:
            logger.error(f"鼠标拖拽失败: {e}")

    def set_speed(self, speed):
        """
        设置鼠标移动速度

        Args:
            speed (float): 速度倍数
        """
        self.speed = max(0.1, min(speed, 5.0))  # 限制在 0.1-5.0 之间
        logger.info(f"鼠标速度已调整为: {self.speed}")

    def get_position(self):
        """获取当前鼠标位置"""
        return pyautogui.position()
