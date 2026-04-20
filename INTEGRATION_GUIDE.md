# 快速集成指南 | Quick Integration Guide

## 中文说明

### 1. 从原项目集成模型

如果你已有训练好的模型（来自 code.ipynb）：

```bash
# 复制模型到项目目录
cp /path/to/nd_model11.keras models/gesture_model.keras

# 更新配置文件中的模型路径
# 编辑 src/config/config.yaml 中的 model.path
```

### 2. 集成数据收集代码

将 code.ipynb 的数据收集部分改造为独立脚本（可选）：

```python
# examples/data_collection_standalone.py
# 可从 code.ipynb 中的数据收集单元提取
```

### 3. 测试系统

```bash
# 测试各个组件
python test_components.py

# 运行完整系统
python main.py
```

### 4. 自定义配置

编辑 `src/config/config.yaml`：
- 修改支持的手势列表
- 调整识别参数
- 配置鼠标/键盘速度

### 5. 扩展功能

#### 添加新手势识别

```python
# 修改配置文件中的 actions
model:
  actions:
    - 'existing_gesture'
    - 'new_gesture_1'  # 新增
    - 'new_gesture_2'  # 新增
```

#### 自定义控制命令

```python
from src.control import CommandExecutor

executor = CommandExecutor()

# 添加自定义手势映射
def my_action(params):
    print("执行自定义操作")
    # 您的代码

executor.register_command('new_gesture', my_action)
```

#### 修改检测参数

在 `src/config/config.yaml` 中调整：
```yaml
recognition:
  min_sequence_length: 10    # 降低识别所需的最少帧数 (更快响应)
  motion_threshold: 0.01     # 调整动作检测敏感度
  pause_time: 2.0            # 识别后的暂停时间
```

## English Guide

### 1. Integrate Pre-trained Model

```bash
# Copy model to project
cp /path/to/your_model.keras models/gesture_model.keras
```

### 2. Test System

```bash
# Test components
python test_components.py

# Run full system
python main.py
```

### 3. Customize Configuration

Edit `src/config/config.yaml`:
- Modify gesture list
- Adjust detection parameters
- Configure control speeds

### 4. Add Custom Gestures

```python
# In src/config/config.yaml
model:
  actions:
    - 'gesture1'
    - 'gesture2'
    - 'your_custom_gesture'  # Add new gesture
```

### 5. Register Custom Commands

```python
from src.control import CommandExecutor

executor = CommandExecutor()

def my_custom_action(params):
    # Your custom logic here
    pass

executor.register_command('my_gesture', my_custom_action)
```

## 文件映射关系

### code.ipynb 代码提取映射

| Notebook 单元 | 提取到 | 文件 |
|---|---|---|
| Cell 3 | mediapipe_detection() | `src/gesture_detector/mediapipe_detector.py` |
| Cell 5 | draw_styled_landmarks() | `src/gesture_detector/mediapipe_detector.py` |
| Cell 13 | extract_keypoints() | `src/gesture_detector/feature_extractor.py` |
| Cell 26-27 | BiLSTM Model | `src/gesture_detector/gesture_classifier.py` |
| Cell 20-21 | Data Collection | (可选) `examples/data_collection.py` |
| Cell 48-49 | Real-time Loop | `main.py` - `process_frame()` |

## 项目结构

```
gesture-recognition-control-system/
├── src/                          # 源代码
│   ├── gesture_detector/         # 检测和识别
│   ├── control/                  # 控制模块
│   ├── utils/                    # 工具函数
│   └── config/                   # 配置文件
├── models/                       # 模型存储
├── data/                         # 数据存储
├── notebooks/                    # Jupyter 参考
├── main.py                       # 主程序
├── test_components.py            # 测试脚本
└── requirements.txt              # 依赖列表
```

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 测试组件
python test_components.py

# 运行系统（使用默认配置）
python main.py

# 运行系统（使用自定义配置）
python main.py --config src/config/config.yaml

# 查看日志
tail -f logs/gesture_control.log
```

## 故障排除

### 问题：ModuleNotFoundError

**解决方案**：确保在项目根目录运行命令

```bash
cd gesture-recognition-control-system
python main.py
```

### 问题：摄像头无法打开

**解决方案**：修改 `camera.device_id`

```yaml
camera:
  device_id: 1  # 尝试不同的 ID (0, 1, 2, ...)
```

### 问题：模型加载失败

**解决方案**：确认模型路径正确

```yaml
model:
  path: 'models/gesture_model.keras'  # 检查文件是否存在
```

### 问题：识别不准确

**解决方案**：调整参数

```yaml
recognition:
  confidence_threshold: 0.85  # 降低置信度阈值（更容易识别）
  min_sequence_length: 8      # 减少所需帧数
  motion_threshold: 0.005     # 提高动作检测灵敏度
```

## 下一步

1. ✅ 安装依赖 → `pip install -r requirements.txt`
2. ✅ 准备模型 → 放入 `models/` 目录
3. ✅ 测试系统 → `python test_components.py`
4. ✅ 运行程序 → `python main.py`
5. ✅ 自定义配置 → 编辑 `src/config/config.yaml`
6. ✅ 扩展功能 → 添加新模块或手势

## 支持

有问题？请查看 README.md 或提交 Issue。
