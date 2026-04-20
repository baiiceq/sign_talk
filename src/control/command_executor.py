"""
命令执行模块
将识别的手势映射到具体的系统操作
"""

import logging
from .mouse_controller import MouseController
from .keyboard_controller import KeyboardController

logger = logging.getLogger(__name__)


class CommandExecutor:
    """
    命令执行器
    根据手势调用相应的控制操作
    """

    def __init__(self):
        """初始化命令执行器"""
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        
        # 手势到命令的映射
        self.gesture_commands = {
            # 鼠标控制手势
            'left_swipe': self._left_swipe,
            'right_swipe': self._right_swipe,
            'up_swipe': self._up_swipe,
            'down_swipe': self._down_swipe,
            'left_click': self._left_click,
            'right_click': self._right_click,
            'double_click': self._double_click,
            'scroll_up': self._scroll_up,
            'scroll_down': self._scroll_down,
            
            # 键盘控制手势
            'copy': self._copy,
            'paste': self._paste,
            'cut': self._cut,
            'undo': self._undo,
            'redo': self._redo,
            'save': self._save,
            'delete': self._delete,
            'enter': self._enter,
            'esc': self._esc,
            
            # 系统控制手势
            'alt_tab': self._alt_tab,
            'switch_window': self._alt_tab,
        }
        
        logger.info("命令执行器已初始化")

    def execute(self, gesture, params=None):
        """
        执行手势对应的命令

        Args:
            gesture (str): 手势名称
            params (dict): 命令参数 (可选)

        Returns:
            bool: 是否执行成功
        """
        if gesture not in self.gesture_commands:
            logger.warning(f"未知的手势命令: {gesture}")
            return False

        try:
            command = self.gesture_commands[gesture]
            command(params or {})
            logger.info(f"执行命令: {gesture}")
            return True
        except Exception as e:
            logger.error(f"执行命令失败 ({gesture}): {e}")
            return False

    def register_command(self, gesture_name, callback):
        """
        注册自定义命令

        Args:
            gesture_name (str): 手势名称
            callback: 回调函数
        """
        self.gesture_commands[gesture_name] = callback
        logger.info(f"已注册自定义命令: {gesture_name}")

    def get_supported_gestures(self):
        """获取所有支持的手势"""
        return list(self.gesture_commands.keys())

    # ==================== 鼠标控制命令 ====================
    
    def _left_swipe(self, params):
        """左滑"""
        self.mouse.move(-50, 0)

    def _right_swipe(self, params):
        """右滑"""
        self.mouse.move(50, 0)

    def _up_swipe(self, params):
        """上滑"""
        self.mouse.move(0, -50)

    def _down_swipe(self, params):
        """下滑"""
        self.mouse.move(0, 50)

    def _left_click(self, params):
        """左键点击"""
        self.mouse.left_click()

    def _right_click(self, params):
        """右键点击"""
        self.mouse.right_click()

    def _double_click(self, params):
        """双击"""
        self.mouse.double_click()

    def _scroll_up(self, params):
        """向上滚动"""
        amount = params.get('amount', 3)
        self.mouse.scroll_up(amount)

    def _scroll_down(self, params):
        """向下滚动"""
        amount = params.get('amount', 3)
        self.mouse.scroll_down(amount)

    # ==================== 键盘控制命令 ====================

    def _copy(self, params):
        """复制"""
        self.keyboard.copy()

    def _paste(self, params):
        """粘贴"""
        self.keyboard.paste()

    def _cut(self, params):
        """剪切"""
        self.keyboard.cut()

    def _undo(self, params):
        """撤销"""
        self.keyboard.undo()

    def _redo(self, params):
        """重做"""
        self.keyboard.redo()

    def _save(self, params):
        """保存"""
        self.keyboard.save()

    def _delete(self, params):
        """删除"""
        self.keyboard.press('delete')

    def _enter(self, params):
        """回车"""
        self.keyboard.press('return')

    def _esc(self, params):
        """退出"""
        self.keyboard.press('esc')

    # ==================== 系统控制命令 ====================

    def _alt_tab(self, params):
        """切换窗口"""
        self.keyboard.toggle_alt_tab()
