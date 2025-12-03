# 扫地机器人地面视角障碍物检测系统 (VLA for IkitBot)

## 🎯 项目概述

基于深度学习的地面视角障碍物检测系统，部署于 Jetson Nano 嵌入式平台，用于扫地机器人实时障碍物识别与避障。

**项目周期**: 7天冲刺开发  
**当前进度**: Day 3 - 模型优化与 Jetson Nano 环境配置完成

## ✨ 功能特性

- 🔍 **多类别障碍物检测**: 电线、拖鞋、袜子、数据线、小玩具等
- ⚡ **实时推理**: 基于 TensorRT 优化，支持 Jetson Nano 端侧部署
- 📷 **多源输入**: 支持 CSI 摄像头、USB 摄像头、RealSense RGBD
- 🔄 **多GPU训练**: 支持 16x RTX 2080 分布式训练

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 训练框架 | PyTorch 2.x + Ultralytics YOLOv8 |
| 模型优化 | ONNX, TensorRT, INT8 量化 |
| 部署平台 | Jetson Nano (JetPack 4.6, CUDA 10.2) |
| 数据格式 | ROS Bag (RGBD) |
| 多GPU训练 | DDP (16x RTX 2080) |

## 📁 项目结构

```
VLA_FOR_IKITBOT/
├── configs/                  # 配置文件
│   └── data.yaml            # YOLO 数据集配置
├── data/                     # 数据目录
│   ├── raw/                 # 原始 ROS Bag 文件
│   ├── extracted_frames/    # 提取的 RGB 帧
│   ├── seed_dataset/        # 种子数据集 (100张)
│   └── yolo_dataset/        # YOLO 格式数据集
├── docs/                     # 文档
│   ├── jetson_setup.md      # Jetson 环境配置
│   ├── gpu_server_guide.md  # 多GPU训练指南
│   └── jetson_nano_test_report.md  # Jetson 测试报告
├── models/                   # 模型文件
│   ├── pretrained/          # 预训练权重
│   └── deployed/            # 部署模型 (ONNX/TensorRT)
├── runs/                     # 训练结果
│   ├── train/               # 正式训练
│   └── validate/            # 验证训练
├── scripts/                  # 自动化脚本
│   ├── train_day2.sh        # Day2 训练脚本
│   ├── train_multi_gpu.sh   # 多GPU训练脚本
│   ├── optimize_day3.sh     # Day3 优化脚本
│   └── setup_jetson.sh      # Jetson 环境配置
├── src/                      # 源代码
│   ├── data_processing/     # 数据处理
│   │   ├── extract_rosbag_images.py
│   │   └── prepare_yolo_dataset.py
│   ├── training/            # 训练脚本
│   │   ├── train_baseline.py
│   │   └── train_distributed.py  # 多GPU训练
│   ├── optimization/        # 模型优化
│   │   └── model_optimization.py
│   └── deployment/          # 部署脚本
│       └── jetson_test.py
└── .github/
    └── scratchpad.md        # 项目进度记录
```

## 🚀 快速开始

### 1. 环境配置

**本地开发环境 (Mac/Linux)**:
```bash
conda create -n obstacle_detection python=3.8
conda activate obstacle_detection
pip install ultralytics opencv-python
```

**GPU 服务器 (16x RTX 2080)**:
```bash
# 参考 docs/gpu_server_guide.md
bash scripts/train_multi_gpu.sh check
```

**Jetson Nano**:
```bash
# 参考 docs/jetson_setup.md
# 或运行一键配置脚本
bash setup_jetson.sh
```

### 2. 数据准备

```bash
# 从 ROS Bag 提取图像
python src/data_processing/extract_rosbag_images.py

# 准备 YOLO 数据集
python src/data_processing/prepare_yolo_dataset.py
```

### 3. 模型训练

```bash
# 单卡训练
bash scripts/train_day2.sh train yolov8n.pt 50

# 多卡训练 (16x RTX 2080)
bash scripts/train_multi_gpu.sh train 100 all
```

### 4. 模型优化

```bash
# 导出 ONNX
python src/optimization/model_optimization.py onnx runs/train/best.pt

# 导出 TensorRT (需要 NVIDIA GPU)
python src/optimization/model_optimization.py tensorrt runs/train/best.pt
```

### 5. Jetson 部署

```bash
# 测试 Jetson 环境
python src/deployment/jetson_test.py all

# 运行推理
python src/deployment/inference.py --model model.engine --source /dev/video0
```

## 📊 开发进度

| 阶段 | 任务 | 状态 |
|------|------|------|
| Day 1 | 环境搭建 & 数据准备 | ✅ 完成 |
| Day 2 | 基线模型训练 | ✅ 流程验证完成 |
| Day 2+ | 多GPU训练支持 | ✅ 完成 |
| Day 3 | 模型优化工具 | ✅ 完成 |
| Day 3+ | Jetson Nano 测试 | ✅ 完成 |
| Day 4 | 端侧部署 | 🔄 进行中 |
| Day 5 | 闭环集成 | ⏳ 待开始 |
| Day 6 | 优化迭代 | ⏳ 待开始 |
| Day 7 | 演示准备 | ⏳ 待开始 |

## 🔧 Jetson Nano 环境状态

| 组件 | 版本 | 状态 |
|------|------|------|
| L4T | R32.7.1 | ✅ |
| CUDA | 10.2 | ✅ |
| cuDNN | 8.2.1 | ✅ |
| TensorRT | 8.2.1 | ✅ |
| PyTorch | 1.10.0 | ✅ |
| OpenCV | 4.1.1 | ✅ |

## 📚 文档

- [Jetson 环境配置指南](docs/jetson_setup.md)
- [多GPU训练指南](docs/gpu_server_guide.md)
- [Jetson Nano 测试报告](docs/jetson_nano_test_report.md)
- [数据标注指南](docs/annotation_guide.md)

## 👥 合作方

**奇勃科技** - 提供 RGBD 数据与硬件支持

## 📄 许可证

MIT License
