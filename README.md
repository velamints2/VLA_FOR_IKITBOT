# 扫地机器人地面视角障碍物检测系统

## 项目概述
基于深度学习的地面视角障碍物检测系统，部署于Jetson Nano嵌入式平台，用于扫地机器人实时障碍物识别与避障。

## 技术栈
- **训练平台**: PyTorch 2.x + Ultralytics YOLO
- **部署平台**: Jetson Nano + TensorRT
- **数据源**: RGBD视频流
- **检测目标**: 电线、拖鞋、小物体等地面障碍物

## 项目结构
```
llm/
├── data/              # 数据目录
│   ├── raw/          # 原始RGBD视频
│   ├── frames/       # 提取的帧（RGB + Depth）
│   ├── annotations/  # 标注文件
│   └── seed_dataset/ # 种子数据集
├── models/           # 模型目录
│   ├── pretrained/   # 预训练权重
│   ├── checkpoints/  # 训练检查点
│   └── deployed/     # 部署模型（ONNX/TensorRT）
├── src/              # 源代码
│   ├── data_processing/  # 数据处理
│   ├── training/         # 训练脚本
│   ├── deployment/       # 部署脚本
│   └── utils/            # 工具函数
├── tests/            # 测试代码
├── docs/             # 文档
├── scripts/          # 自动化脚本
└── configs/          # 配置文件
```

## 快速开始

### 环境配置
```bash
# 服务器训练环境
conda create -n obstacle_detection python=3.8
conda activate obstacle_detection
pip install -r requirements.txt

# Jetson Nano部署环境
# 参考 docs/jetson_setup.md
```

### 训练
```bash
python src/training/train.py --config configs/yolo_config.yaml
```

### 部署
```bash
python src/deployment/deploy_jetson.py --model models/deployed/model.trt
```

## 开发进度
- [x] 第1天：环境搭建与数据准备
- [ ] 第2天：基线模型训练
- [ ] 第3天：模型轻量化
- [ ] 第4天：端侧部署
- [ ] 第5天：闭环集成
- [ ] 第6天：优化迭代
- [ ] 第7天：演示准备

## 合作方
奇勃科技 - 提供RGBD数据与硬件支持

## 许可证
MIT License
