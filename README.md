# 扫地机器人地面视角障碍物检测系统 (VLA for IkitBot)

用于扫地机器人地面视角障碍物检测的端到端方案，覆盖数据采集、半自动标注、分布式训练、模型轻量化、Jetson Nano 端侧验证与部署。核心目标是 7 天内交付可在嵌入式设备实时运行的视觉感知模块。

**当前里程碑**: Day 3 完成模型优化与 Jetson 环境验证，分布式训练与半自动标注已可用。

## 🎯 亮点与成果
- 多类别检测：电线、拖鞋、袜子、数据线、小玩具等小目标
- 半自动标注：YOLO11n 预标注 + Label Studio 协作，效率预计提升 70%
- 训练到部署一键脚本：数据提取、训练、优化、Jetson 验证均有现成脚本
- 轻量化与导出：支持 ONNX、(待 GPU) TensorRT 与 INT8 量化
- 端侧适配：Jetson Nano (JetPack 4.6, CUDA 10.2) 环境与基准测试报告已完成

## 🧭 系统架构概览
- **数据与标注**：ROS Bag → 帧提取 → 种子集筛选 → 半自动标注 → YOLO 数据集 ([src/data_processing](src/data_processing))
- **训练**：单卡/多卡 (DDP) 训练脚本，自动计算批大小，支持混合精度 ([src/training](src/training))
- **模型优化**：剪枝、量化、导出 ONNX/TensorRT、性能基准 ([src/optimization/model_optimization.py](src/optimization/model_optimization.py))
- **部署与验证**：Jetson 环境检测与性能测试 ([src/deployment/jetson_test.py](src/deployment/jetson_test.py))
- **标注后台**：Label Studio + 自定义 ML Backend ([yolo_backend](yolo_backend))，配合自动/协作标注脚本
- **自动化脚本**：数据准备、训练、优化、标注、环境配置等一键脚本集中于 [scripts](scripts)

## 📁 目录速览
- [configs](configs) 配置文件（YOLO 数据集、类别等）
- [data](data) 原始 Bag、提取帧、种子集、增广与训练数据
- [docs](docs) 环境、训练、部署、标注等操作文档
- [label_studio](label_studio) 标注前端配置与使用指南
- [models](models) 预训练、部署模型与检查点
- [runs](runs) 训练与验证输出
- [scripts](scripts) 数据、训练、优化、标注、环境一键脚本
- [src](src) 核心代码：数据处理、训练、优化、部署
- [yolo_backend](yolo_backend) Label Studio ML Backend 服务

示例子目录（数据与代码主干）：
```
src/
├── data_processing/   # Bag 转帧、数据筛选、切分、标注校验
├── training/          # train.py、train_baseline.py、train_distributed.py
├── optimization/      # model_optimization.py（ONNX/量化/基准）
└── deployment/        # jetson_test.py（端侧验证）

scripts/
├── train_day2.sh          # 单卡训练
├── train_multi_gpu.sh     # DDP 多卡训练
├── optimize_day3.sh       # 模型优化与导出
├── auto_annotate.sh /.py  # 半自动预标注
├── start_label_studio.sh  # 启动 Label Studio 前端
├── setup_jetson.sh        # Jetson 环境配置
└── benchmark_jetson.py    # 端侧基准测试
```

## 🚀 快速上手

### 1) 环境准备
- 本地/单GPU：建议 Python 3.8+，安装 ultralytics、opencv-python、torch（GPU 版）
- 多 GPU 服务器：参考 [docs/gpu_server_guide.md](docs/gpu_server_guide.md)，可先运行 `bash scripts/train_multi_gpu.sh check`
- Jetson Nano：参考 [docs/jetson_setup.md](docs/jetson_setup.md) 或一键脚本 `bash scripts/setup_jetson.sh`

### 2) 数据流程（Bag → YOLO 数据集）
```bash
# 从 ROS Bag 批量抽帧
python src/data_processing/extract_rosbag_images.py --batch data/raw data/frames 2

# 选择代表性帧作为种子集
python src/data_processing/select_seed_dataset.py data/frames --output data/seed_dataset_v2 --num 200

# 半自动预标注（YOLO11n 预标 + 可视化）
bash scripts/auto_annotate.sh data/seed_dataset_v2

# YOLO 数据集划分
python src/data_processing/split_dataset.py data/seed_dataset_v2 --train-ratio 0.8
```

### 3) 标注协作（可选）
- 快速本地：LabelImg 配合预标注结果
- 团队协作：
  ```bash
  bash scripts/start_label_studio.sh    # 启动前端
  # 浏览器打开 http://localhost:8080，连接 ML Backend（默认 http://localhost:9090）
  ```
- 详见 [label_studio/README.md](label_studio/README.md) 与 [docs/annotation_tools_guide.md](docs/annotation_tools_guide.md)

### 4) 训练
```bash
# 单卡/本地快速训练
bash scripts/train_day2.sh train yolo11n.pt 50

# 多卡分布式训练（自动批大小、AMP）
bash scripts/train_multi_gpu.sh train 100 all
```

### 5) 模型优化与导出
```bash
# ONNX 导出 + 基准
python src/optimization/model_optimization.py onnx runs/train/best.pt

# TensorRT / INT8（需 NVIDIA GPU）
python src/optimization/model_optimization.py tensorrt runs/train/best.pt
```

### 6) Jetson 验证
```bash
# 环境/性能自检（摄像头、CUDA、延迟等）
python src/deployment/jetson_test.py all
```

## 📚 文档索引
- 环境与硬件：[docs/jetson_setup.md](docs/jetson_setup.md)｜[docs/gpu_server_guide.md](docs/gpu_server_guide.md)
- 数据与提取：[docs/data_extraction_report.md](docs/data_extraction_report.md)｜[docs/extract_bag_on_macos.md](docs/extract_bag_on_macos.md)
- 标注与协作：[docs/annotation_tools_guide.md](docs/annotation_tools_guide.md)｜[docs/annotation_guide.md](docs/annotation_guide.md)｜[label_studio/README.md](label_studio/README.md)
- 优化与部署：[docs/jetson_nano_test_report.md](docs/jetson_nano_test_report.md)

## 📊 里程碑状态
- Day 1 环境与数据就绪：✅ 完成
- Day 2 基线训练：✅ 流程验证完成
- Day 2+ 多 GPU 训练：✅ 完成
- Day 3 模型优化工具：✅ 完成
- Day 3+ Jetson Nano 测试：✅ 完成
- Day 4 端侧部署：🔄 进行中
- Day 5 闭环集成：⏳ 待开始
- Day 6 优化迭代：⏳ 待开始
- Day 7 演示准备：⏳ 待开始

## 👥 合作方
奇勃科技（数据与硬件支持）

## 📄 许可证
MIT License
