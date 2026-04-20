# 手部关键点检测与机器学习的实时手势识别与交互控制系统

一个基于 **MediaPipe** 和 **BiLSTM** 神经网络的实时手势识别与交互控制系统。通过摄像头实时检测手部关键点，使用深度学习模型识别手势，支持鼠标和键盘控制。

## 📋 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [扩展开发](#扩展开发)

## ✨ 功能特性

### 核心功能

- ✅ **实时手部检测** - 使用 MediaPipe 进行实时人体关键点检测
- ✅ **手势识别** - 基于 BiLSTM 的序列化手势识别
- ✅ **交互控制** - 支持鼠标和键盘自动化控制
- ✅ **多手势支持** - 可扩展的手势库 (默认10种印地手势)
- ✅ **实时可视化** - 展示检测结果、置信度、手势历史

### 控制功能

| 类别 | 操作 | 说明 |
|------|------|------|
| **鼠标** | 移动、点击、滚动、拖拽 | 支持左键、右键、双击 |
| **键盘** | 输入、快捷键、系统命令 | Ctrl+C/V, Alt+Tab 等 |
| **系统** | 窗口切换、应用控制 | 支持自定义命令 |

### 高级特性

- 🎯 **动作检测** - 仅在有明显动作时才进行识别
- 🔄 **暂停机制** - 识别后自动暂停，防止重复触发
- 📊 **实时统计** - FPS显示、置信度展示
- 🔧 **灵活配置** - YAML 配置文件，支持自定义参数
- 📝 **完整日志** - 详细的事件记录和调试信息

## 🏗️ 系统架构

```
┌─────────────────────────────────────────┐
│     摄像头输入 (Video Stream)            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    MediaPipe 检测器                      │
│  (身体、手部关键点)                       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    特征提取器                             │
│  (空间坐标、几何特征)                     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    手势分类器 (BiLSTM)                   │
│  (序列识别、置信度过滤)                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    命令执行器                             │
│  (鼠标、键盘、系统控制)                   │
└─────────────────────────────────────────┘
```

## 📦 环境要求

- Python 3.7+
- Windows / macOS / Linux
- 摄像头设备
- 显卡 (推荐 GPU 加速推理)

## 🚀 安装指南

### 1. 克隆项目

```bash
git clone <repository-url>
cd gesture-recognition-control-system
```

### 2. 创建虚拟环境 (推荐)

```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备模型文件

将训练好的 BiLSTM 模型放在 `models/` 目录：

```bash
cp path/to/gesture_model.keras models/
```

## 🎯 快速开始

### 基本运行

```bash
# 使用默认配置
python main.py

# 使用自定义配置文件
python main.py --config my_config.yaml
```

### 交互操作

- **退出** - 按 'q' 键或 Ctrl+C

## ⚙️ 配置说明

编辑 `src/config/config.yaml` 进行配置：

### 模型配置

```yaml
model:
  path: 'models/gesture_model.keras'
  sequence_length: 30                    # 单个序列的帧数
  confidence_threshold: 0.90             # 预测置信度阈值
```

### 摄像头配置

```yaml
camera:
  device_id: 0                           # 摄像头ID
  frame_width: 640
  frame_height: 480
  fps: 30
```

### 识别参数

```yaml
recognition:
  min_sequence_length: 10                # 最小帧数
  motion_threshold: 0.01                 # 动作检测灵敏度
  pause_time: 2.0                        # 识别暂停时间
```

### 控制配置

```yaml
control:
  mouse_speed: 1.0                       # 鼠标速度倍数
  keyboard_interval: 0.05                # 键盘输入间隔
  enable_mouse: true
  enable_keyboard: true
```

## 📁 项目结构

```
gesture-recognition-control-system/
├── src/
│   ├── gesture_detector/               # 手势检测模块
│   │   ├── __init__.py
│   │   ├── mediapipe_detector.py       # MediaPipe 封装
│   │   ├── feature_extractor.py        # 特征提取
│   │   └── gesture_classifier.py       # 分类器
│   │
│   ├── control/                        # 交互控制模块
│   │   ├── __init__.py
│   │   ├── mouse_controller.py         # 鼠标控制
│   │   ├── keyboard_controller.py      # 键盘控制
│   │   └── command_executor.py         # 命令执行
│   │
│   ├── utils/                          # 工具模块
│   │   ├── __init__.py
│   │   ├── logger_config.py            # 日志配置
│   │   └── config_loader.py            # 配置加载
│   │
│   ├── config/                         # 配置文件
│   │   ├── __init__.py
│   │   └── config.yaml                 # 主配置文件
│   │
│   └── __init__.py
│
├── models/                             # 模型存放目录
│   └── gesture_model.keras             # 训练好的模型
│
├── data/                               # 数据目录
│   ├── gestures/                       # 手势样本数据
│   └── training_logs/                  # 训练日志
│
├── notebooks/                          # Jupyter 笔记本
│   ├── data_collection.ipynb           # 数据收集
│   ├── model_training.ipynb            # 模型训练
│   └── demo.ipynb                      # 演示
│
├── logs/                               # 程序运行日志
│
├── main.py                             # 主程序入口
├── requirements.txt                    # 依赖列表
└── README.md                           # 本文件
```

## 🔨 扩展开发

### 添加自定义手势

1. **修改配置文件**

```yaml
model:
  actions:
    - 'gesture1'
    - 'gesture2'
    - 'your_custom_gesture'
```

2. **收集训练数据** (使用提供的 notebook)

3. **重新训练模型**

### 注册自定义命令

```python
from src.control import CommandExecutor

executor = CommandExecutor()

# 注册自定义命令
def my_custom_action(params):
    print("执行自定义操作")

executor.register_command('my_gesture', my_custom_action)
```

### 自定义检测参数

```python
from src.gesture_detector import MediaPipeDetector

detector = MediaPipeDetector(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
```

## 📊 性能优化建议

- 使用 GPU 加速 (配置 TensorFlow GPU)
- 调整 `motion_threshold` 提高识别速度
- 减少 `sequence_length` 加快响应
- 增加 `pause_time` 减少误触发

## 🐛 常见问题

### Q: 摄像头无法打开？
A: 检查设备 ID，尝试修改 `camera.device_id` 为 1 或 2

### Q: 识别不准确？
A: 调整 `confidence_threshold` 或重新训练模型

### Q: 控制命令没有响应？
A: 检查 `control.enable_mouse/keyboard` 是否启用

## 📝 数据标注与训练流程

参考 `notebooks/` 目录中的 Jupyter 笔记本：

1. **数据收集** - `data_collection.ipynb`
2. **模型训练** - `model_training.ipynb`  
3. **演示测试** - `demo.ipynb`

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 👤 作者

基于 Real-Time-Gesture-Recognition-Using-Mediapipe-and-BiLSTM 扩展

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 📧 联系方式

有问题或建议？请提交 GitHub Issue。

---

**快速开始命令：**

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备模型文件
cp /path/to/model models/gesture_model.keras

# 3. 运行程序
python main.py
```

**按 'q' 键退出程序**
