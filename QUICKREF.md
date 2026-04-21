# 项目概览和快速参考

## 🎯 项目一句话总结

基于 MediaPipe + BiLSTM 的实时手势识别系统，支持将手势映射到鼠标/键盘操作，实现无接触交互控制。

---

## 📊 核心模块速查表

### 1. 手势检测模块 (`src/gesture_detector/`)

| 文件 | 类 | 功能 |
|------|-----|------|
| mediapipe_detector.py | `MediaPipeDetector` | 实时关键点检测 |
| feature_extractor.py | `FeatureExtractor` | 特征向量提取 |
| gesture_classifier.py | `GestureClassifier` | 手势分类预测 |

**使用示例：**
```python
from src.gesture_detector import MediaPipeDetector, FeatureExtractor

detector = MediaPipeDetector()
extractor = FeatureExtractor()

image, results = detector.detect(frame)
features = extractor.extract(results)
```

### 2. 控制模块 (`src/control/`)

| 文件 | 类 | 功能 |
|------|-----|------|
| mouse_controller.py | `MouseController` | 鼠标移动、点击、滚动 |
| keyboard_controller.py | `KeyboardController` | 按键、快捷键、文本输入 |
| command_executor.py | `CommandExecutor` | 手势到命令的映射执行 |

**使用示例：**
```python
from src.control import CommandExecutor

executor = CommandExecutor()
executor.execute('left_click')
executor.execute('copy')
```

### 3. 工具模块 (`src/utils/`)

| 文件 | 类/函数 | 功能 |
|------|--------|------|
| logger_config.py | `setup_logger()` | 日志配置 |
| config_loader.py | `ConfigLoader` | YAML/JSON 配置加载 |

**使用示例：**
```python
from src.utils import setup_logger, ConfigLoader

logger = setup_logger('myapp')
config = ConfigLoader('config.yaml')
model_path = config.get('model.path')
```

---

## 🔄 数据流程

```
视频帧
  ↓
MediaPipeDetector.detect()
  ├→ 颜色转换 (BGR → RGB)
  ├→ 关键点检测
  └→ 颜色转换 (RGB → BGR)
  ↓
FeatureExtractor.extract()
  ├→ 提取姿态特征 (132维)
  ├→ 提取单手特征 (63维)
  └→ 计算几何特征 (5维)
  ↓
GestureClassifier.predict()
  ├→ 序列长度检查
  ├→ BiLSTM 推理
  └→ 置信度过滤
  ↓
CommandExecutor.execute()
  ├→ 鼠标控制
  ├→ 键盘控制
  └→ 系统命令
```

### 数据采集完成后的下一步（直接执行）

```bash
python scripts/train_model.py \
  --data-dir data/gestures \
  --output-dir models \
  --model-name gesture_model \
  --sequence-length 30 \
  --feature-dim 200 \
  --epochs 40 \
  --batch-size 64
```

产物：
- `models/gesture_model.pt`（运行时默认模型）
- `models/gesture_model.ts`（TorchScript）
- `models/gesture_model_labels.json`（标签索引）
- `models/gesture_model_metrics.json`（训练指标）

---

## ⚙️ 快速配置修改

### 修改手势列表

编辑 `src/config/config.yaml`:
```yaml
model:
  actions:
    - 'new_gesture1'
    - 'new_gesture2'
```

### 调整识别灵敏度

```yaml
recognition:
  confidence_threshold: 0.85  # 降低更容易识别
  motion_threshold: 0.005     # 提高动作检测灵敏度
  min_sequence_length: 8      # 减少所需帧数，更快响应
```

### 修改摄像头

```yaml
camera:
  device_id: 1  # 尝试其他 ID
```

---

## 🚀 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 测试系统
python test_components.py

# 运行主程序
python main.py

# 查看日志
tail -f logs/gesture_control.log

# 运行自定义配置
python main.py --config my_config.yaml
```

---

## 📌 关键参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `confidence_threshold` | 0.90 | 0-1 | 预测置信度阈值 |
| `min_sequence_length` | 10 | 1-30 | 最小序列长度 |
| `motion_threshold` | 0.01 | 0.001-0.1 | 动作检测敏感度 |
| `pause_time` | 2.0 | 0.5-10 | 识别后暂停时间 |
| `mouse_speed` | 1.0 | 0.1-5.0 | 鼠标速度倍数 |

---

## 🎮 支持的控制操作

### 鼠标操作
- `left_click` - 左键点击
- `right_click` - 右键点击
- `double_click` - 双击
- `scroll_up` - 向上滚动
- `scroll_down` - 向下滚动
- `left_swipe` - 左滑
- `right_swipe` - 右滑

### 键盘操作
- `copy` - Ctrl+C
- `paste` - Ctrl+V
- `cut` - Ctrl+X
- `undo` - Ctrl+Z
- `redo` - Ctrl+Y
- `save` - Ctrl+S
- `delete` - Delete 键
- `enter` - 回车
- `esc` - Esc 键

### 系统操作
- `alt_tab` - 窗口切换
- `switch_window` - 同上

---

## 💡 开发建议

### 性能优化
1. 使用 GPU 加速 (PyTorch + CUDA)
2. 降低 `min_sequence_length` 加快响应
3. 增加 `motion_threshold` 减少计算
4. 降低视频分辨率 (如 320x240)

### 准确度提升
1. 收集更多训练数据
2. 调整 `confidence_threshold` 
3. 增加 `min_sequence_length`
4. 使用数据增强技术

### 可靠性改进
1. 添加错误处理
2. 实现暂停/恢复机制
3. 添加用户反馈机制
4. 记录详细日志

---

## 📚 文件对应关系

| 功能 | 对应文件 | 关键方法/类 |
|------|---------|-----------|
| 摄像头输入 | main.py | `cv2.VideoCapture` |
| 关键点检测 | mediapipe_detector.py | `MediaPipeDetector.detect()` |
| 特征提取 | feature_extractor.py | `FeatureExtractor.extract()` |
| 手势分类 | gesture_classifier.py | `GestureClassifier.predict()` |
| 鼠标控制 | mouse_controller.py | `MouseController.move()` |
| 键盘控制 | keyboard_controller.py | `KeyboardController.press()` |
| 命令映射 | command_executor.py | `CommandExecutor.execute()` |
| 配置管理 | config_loader.py | `ConfigLoader.get()` |

---

## 🔧 扩展示例

### 添加新的控制命令

```python
# 在 CommandExecutor 中添加
def _my_custom_action(self, params):
    """自定义操作"""
    self.keyboard.hotkey('ctrl', 'alt', 'delete')

# 注册命令
executor.register_command('my_gesture', self._my_custom_action)
```

### 自定义特征提取

```python
# 在 FeatureExtractor 中扩展
def _extract_custom_features(self, results):
    """提取自定义特征"""
    # 您的代码
    pass
```

### 集成额外检测器

```python
# 结合其他检测库
from mediapipe import solutions

# 添加新的检测管道
```

---

## 📞 故障排除

**问题：导入错误**
- 检查 `src/__init__.py` 是否存在
- 确保从项目根目录运行

**问题：模型加载失败**
- 验证模型文件路径
- 检查文件格式是否正确 (.pt)

**问题：识别不工作**
- 检查摄像头是否正常
- 调整 `confidence_threshold`

**问题：控制命令无反应**
- 确认 `enable_mouse/keyboard` 已启用
- 检查应用程序是否获得焦点

---

## 📖 相关文档

- [README.md](README.md) - 完整项目文档
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - 集成指南
- [src/config/config.yaml](src/config/config.yaml) - 配置参考

---

**最后修改**: 2024年
**版本**: 1.0.0
