# ROS Demo - 扫地机器人障碍物检测系统

本目录包含用于演示扫地机器人障碍物检测系统的 ROS1 节点。

## 📁 目录结构

```
ros_demo/
├── README.md                    # 本文档
├── __init__.py                  # Python 包初始化
├── obstacle_detector_node.py    # 实时检测节点
└── video_processor_node.py      # 离线视频处理节点
```

## 🚀 快速开始

### 方式一：独立模式（无需ROS）

如果您没有安装 ROS，可以使用独立模式运行：

```bash
# 使用摄像头实时检测
python src/ros_demo/obstacle_detector_node.py --standalone --source 0

# 使用视频文件
python src/ros_demo/obstacle_detector_node.py --standalone --source path/to/video.mp4

# 指定模型和置信度
python src/ros_demo/obstacle_detector_node.py --standalone --model models/best.pt --conf 0.6
```

### 方式二：视频文件处理

```bash
# 处理单个视频
python src/ros_demo/video_processor_node.py --input demo.mp4 --output demo_annotated.mp4

# 批量处理目录
python src/ros_demo/video_processor_node.py --input videos/ --output results/

# 仅生成统计报告
python src/ros_demo/video_processor_node.py --input demo.mp4 --stats-only --report stats.md
```

### 方式三：ROS 模式

```bash
# 1. 启动 ROS Master
roscore

# 2. 启动完整 Demo（实时摄像头）
roslaunch obstacle_detection demo.launch realtime:=true

# 3. 或使用视频文件
roslaunch obstacle_detection demo.launch realtime:=false video_path:=/path/to/video.mp4
```

## 📦 安装依赖

### Python 依赖

```bash
pip install -r requirements_ros.txt
```

### ROS 包依赖（Ubuntu）

```bash
# ROS Noetic (Ubuntu 20.04)
sudo apt update
sudo apt install ros-noetic-usb-cam ros-noetic-image-view ros-noetic-cv-bridge ros-noetic-rqt-image-view

# ROS Melodic (Ubuntu 18.04)
sudo apt install ros-melodic-usb-cam ros-melodic-image-view ros-melodic-cv-bridge ros-melodic-rqt-image-view
```

### 编译 ROS 包

```bash
# 进入 catkin 工作空间
cd ~/catkin_ws/src

# 链接项目（或复制）
ln -s /path/to/llm obstacle_detection

# 编译
cd ~/catkin_ws
catkin_make

# 刷新环境
source devel/setup.bash
```

## 🔧 节点说明

### obstacle_detector_node.py

**功能**：订阅图像话题，使用 YOLO 模型进行实时障碍物检测，发布标注结果。

**订阅话题**：
- `/camera/image_raw` (sensor_msgs/Image) - 输入图像

**发布话题**：
- `/obstacle_detection/result` (sensor_msgs/Image) - 标注后的图像
- `/obstacle_detection/info` (std_msgs/String) - 检测信息

**参数**：
| 参数 | 默认值 | 描述 |
|------|--------|------|
| `~model_path` | `models/best.pt` | YOLO 模型路径 |
| `~conf_threshold` | `0.5` | 置信度阈值 |
| `~device` | `cpu` | 推理设备 (cpu/cuda/mps) |
| `~imgsz` | `640` | 输入图像大小 |

**命令行参数**：
```bash
python obstacle_detector_node.py --help

# 独立模式
python obstacle_detector_node.py --standalone --source 0 --conf 0.5

# ROS 模式
rosrun obstacle_detection obstacle_detector_node.py _conf_threshold:=0.6
```

### video_processor_node.py

**功能**：处理本地视频文件，生成标注视频和统计报告。

**命令行参数**：
```bash
python video_processor_node.py --help

# 处理视频
python video_processor_node.py -i video.mp4 -o output.mp4

# 批量处理 + 报告
python video_processor_node.py -i videos/ -o results/ -r report.md
```

## 🚀 Launch 文件

### demo.launch

完整的演示启动文件，支持实时和视频文件两种模式。

```bash
# 实时模式
roslaunch obstacle_detection demo.launch realtime:=true

# 视频文件模式
roslaunch obstacle_detection demo.launch realtime:=false video_path:=/path/to/video.mp4

# 自定义参数
roslaunch obstacle_detection demo.launch \
    realtime:=true \
    camera_device:=/dev/video1 \
    conf_threshold:=0.6 \
    device:=cuda
```

### camera_only.launch

仅启动摄像头，用于测试。

```bash
roslaunch obstacle_detection camera_only.launch device:=/dev/video0
```

### detector_only.launch

仅启动检测节点（假设已有图像话题）。

```bash
roslaunch obstacle_detection detector_only.launch input_topic:=/my_camera/image
```

## 📊 检测类别

当前模型支持检测以下障碍物类别：

| ID | 类别 | 描述 |
|----|------|------|
| 0 | wire | 电线/线缆 |
| 1 | shoe | 鞋子 |
| 2 | small_object | 小物体 |

## 💡 性能优化建议

### CPU 优化

```python
# 降低输入分辨率
node.imgsz = 320  # 从 640 降到 320

# 降低置信度阈值（减少后处理）
node.conf_threshold = 0.7
```

### GPU 优化 (CUDA)

```bash
# 使用 CUDA 推理
python obstacle_detector_node.py --device cuda

# 或通过 ROS 参数
rosrun obstacle_detection obstacle_detector_node.py _device:=cuda
```

### Jetson Nano 优化

1. 使用 TensorRT 引擎：
```bash
python src/optimization/model_optimization.py tensorrt models/best.pt --half
```

2. 启用 Jetson Clocks：
```bash
sudo jetson_clocks
```

3. 使用优化的模型：
```bash
python obstacle_detector_node.py --model models/best.engine
```

## 🐛 常见问题

### Q: 找不到摄像头

```bash
# 检查设备
ls /dev/video*

# 测试摄像头
v4l2-ctl --list-devices
```

### Q: cv_bridge 导入错误

```bash
# 安装 cv_bridge
sudo apt install ros-noetic-cv-bridge python3-cv-bridge
```

### Q: 模型加载失败

```bash
# 检查模型文件
ls -la models/best.pt

# 验证模型
python -c "from ultralytics import YOLO; YOLO('models/best.pt')"
```

### Q: 推理速度慢

1. 检查是否在使用 CPU
2. 降低输入分辨率
3. 使用 ONNX 或 TensorRT 优化模型

## 📝 更新日志

- **v1.0.0** - 初始版本
  - 实时检测节点
  - 视频处理节点
  - ROS Launch 文件
  - 独立模式支持

