"""
配置加载模块
支持 YAML 配置文件
"""

import yaml
import json
import os
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    配置文件加载器
    支持 YAML 和 JSON 格式
    """

    def __init__(self, config_file=None):
        """
        初始化配置加载器

        Args:
            config_file (str): 配置文件路径
        """
        self.config_file = config_file
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            self.load(config_file)

    def load(self, config_file):
        """
        加载配置文件

        Args:
            config_file (str): 配置文件路径

        Returns:
            dict: 加载的配置
        """
        try:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            elif config_file.endswith('.json'):
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"不支持的文件格式: {config_file}")
                return {}

            logger.info(f"配置文件已加载: {config_file}")
            return self.config

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    def get(self, key, default=None):
        """
        获取配置值

        Args:
            key (str): 配置键 (支持嵌套访问，用 '.' 分隔)
            default: 默认值

        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key, value):
        """
        设置配置值

        Args:
            key (str): 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config

        # 创建嵌套字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, config_file=None):
        """
        保存配置文件

        Args:
            config_file (str): 保存文件路径
        """
        config_file = config_file or self.config_file

        if not config_file:
            logger.error("未指定配置文件路径")
            return False

        try:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            elif config_file.endswith('.json'):
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)

            logger.info(f"配置文件已保存: {config_file}")
            return True

        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False

    def get_all(self):
        """获取所有配置"""
        return self.config

    def update(self, config_dict):
        """
        更新配置

        Args:
            config_dict (dict): 配置字典
        """
        self.config.update(config_dict)
