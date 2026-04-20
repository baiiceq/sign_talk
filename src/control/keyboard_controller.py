"""
键盘控制模块
通过手势控制系统键盘输入
"""

import pyautogui
import logging

logger = logging.getLogger(__name__)

# 禁用故障保全
pyautogui.FAIL_SAFE = False


class KeyboardController:
    """
    键盘控制器
    支持按键、文本输入、快捷键等操作
    """

    # 常用按键映射
    KEY_MAP = {
        'enter': 'return',
        'space': 'space',
        'tab': 'tab',
        'backspace': 'backspace',
        'delete': 'delete',
        'home': 'home',
        'end': 'end',
        'pageup': 'pageup',
        'pagedown': 'pagedown',
        'left': 'left',
        'right': 'right',
        'up': 'up',
        'down': 'down',
        'esc': 'esc',
        'f1': 'f1',
        'f2': 'f2',
        'f3': 'f3',
        'f4': 'f4',
        'f5': 'f5',
        'f6': 'f6',
        'f7': 'f7',
        'f8': 'f8',
        'f9': 'f9',
        'f10': 'f10',
        'f11': 'f11',
        'f12': 'f12',
    }

    def __init__(self, type_interval=0.05):
        """
        初始化键盘控制器

        Args:
            type_interval (float): 输入文本的间隔时间 (秒)
        """
        self.type_interval = type_interval
        logger.info(f"键盘控制器已初始化 (输入间隔: {type_interval}秒)")

    def press(self, key):
        """
        按下按键

        Args:
            key (str): 按键名称
        """
        try:
            # 首先检查是否在 KEY_MAP 中
            key = self.KEY_MAP.get(key.lower(), key.lower())
            pyautogui.press(key)
            logger.debug(f"按键: {key}")
        except Exception as e:
            logger.error(f"按键失败: {e}")

    def type_text(self, text):
        """
        输入文本

        Args:
            text (str): 要输入的文本
        """
        try:
            pyautogui.typewrite(text, interval=self.type_interval)
            logger.debug(f"输入文本: {text}")
        except Exception as e:
            logger.error(f"文本输入失败: {e}")

    def type_unicode(self, text):
        """
        输入 Unicode 文本 (支持中文等)

        Args:
            text (str): 要输入的文本
        """
        try:
            pyautogui.write(text)
            logger.debug(f"输入 Unicode 文本: {text}")
        except Exception as e:
            # 如果 write 不支持，尝试使用剪贴板
            try:
                import pyperclip
                pyperclip.copy(text)
                pyautogui.hotkey('ctrl', 'v')
                logger.debug(f"通过剪贴板输入: {text}")
            except Exception as e2:
                logger.error(f"文本输入失败: {e2}")

    def hotkey(self, *keys):
        """
        按下快捷键组合

        Args:
            keys: 按键序列 (例如: hotkey('ctrl', 'c'))
        """
        try:
            pyautogui.hotkey(*keys)
            logger.debug(f"快捷键: {' + '.join(keys)}")
        except Exception as e:
            logger.error(f"快捷键失败: {e}")

    def copy(self):
        """复制 (Ctrl+C)"""
        self.hotkey('ctrl', 'c')

    def paste(self):
        """粘贴 (Ctrl+V)"""
        self.hotkey('ctrl', 'v')

    def cut(self):
        """剪切 (Ctrl+X)"""
        self.hotkey('ctrl', 'x')

    def undo(self):
        """撤销 (Ctrl+Z)"""
        self.hotkey('ctrl', 'z')

    def redo(self):
        """重做 (Ctrl+Y)"""
        self.hotkey('ctrl', 'y')

    def save(self):
        """保存 (Ctrl+S)"""
        self.hotkey('ctrl', 's')

    def select_all(self):
        """全选 (Ctrl+A)"""
        self.hotkey('ctrl', 'a')

    def toggle_alt_tab(self):
        """切换应用窗口 (Alt+Tab)"""
        self.hotkey('alt', 'tab')

    def open_run_dialog(self):
        """打开运行对话框 (Win+R)"""
        self.hotkey('win', 'r')

    def set_type_interval(self, interval):
        """设置输入间隔"""
        self.type_interval = interval
        logger.info(f"输入间隔已调整为: {interval}秒")
