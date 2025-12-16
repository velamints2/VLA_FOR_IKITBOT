# 16x RTX 2080 GPU 服务器训练指南

## 🖥️ 服务器配置

| 组件 | 配置 |
|------|------|
| GPU | 16x NVIDIA RTX 2080 (8GB each) |
| 总显存 | 128 GB |
| 推荐Batch | 256 (16 GPU × 16 per GPU) |

## 🚀 快速开始

### 1. 上传代码到服务器

```bash
# 在本地执行
rsync -avz --exclude='data/raw' --exclude='runs/' \
    /Users/macbookair/Documents/trae_projects/llm/ \
    user@gpu-server:/path/to/project/

# 或使用 scp
scp -r llm/ user@gpu-server:/path/to/project/
```

### 2. 服务器环境设置

```bash
# SSH 到服务器
ssh user@gpu-server

# 进入项目目录
cd /path/to/project/llm

# 检查 GPU
nvidia-smi

# 创建 conda 环境
conda create -n obstacle_detection python=3.8 -y
conda activate obstacle_detection

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install ultralytics opencv-python-headless tensorboard

# 验证环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 3. 检查多GPU环境

```bash
python src/training/train_distributed.py check
```

预期输出：
```
======================================================================
多GPU训练环境检查 (16x RTX 2080 Server)
======================================================================
✓ PyTorch: 2.x.x
✓ CUDA: 11.8

发现 16 个 GPU:
  GPU 0: NVIDIA GeForce RTX 2080 (8.0 GB, SM 7.5)
  GPU 1: NVIDIA GeForce RTX 2080 (8.0 GB, SM 7.5)
  ...
  GPU 15: NVIDIA GeForce RTX 2080 (8.0 GB, SM 7.5)

总显存: 128.0 GB
✓ NCCL 后端可用 (推荐用于多GPU)
✓ Ultralytics: 8.x.x
======================================================================
```

### 4. 上传数据集

```bash
# 上传标注好的数据集
rsync -avz data/yolo_dataset/ user@gpu-server:/path/to/project/llm/data/yolo_dataset/
```

### 5. 开始训练

```bash
# 使用所有16张GPU
bash scripts/train_multi_gpu.sh train 100 all

# 或直接使用 Python
python src/training/train_distributed.py ddp \
    --data configs/data.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --gpus all \
    --amp \
    --cache
```

## ⚡ 训练配置推荐

### RTX 2080 (8GB) 最优配置

| 图像大小 | 单卡Batch | 16卡总Batch | 显存占用 |
|----------|-----------|-------------|----------|
| 320×320 | 32 | 512 | ~4GB |
| 480×480 | 24 | 384 | ~5GB |
| 640×640 | 16 | 256 | ~6GB |
| 800×800 | 8 | 128 | ~7GB |

### 大Batch训练技巧

```python
# 学习率线性缩放
# base_lr = 0.01 (batch=64)
# 实际 lr = base_lr * (actual_batch / 64)
lr0 = 0.01 * (256 / 64)  # = 0.04

# 使用 AdamW 优化器（大batch更稳定）
optimizer = 'AdamW'

# 增加 warmup
warmup_epochs = 5

# 使用更强的数据增强
mixup = 0.2
mosaic = 1.0
```

## 📊 监控训练

### TensorBoard

```bash
# 在服务器上启动 TensorBoard
tensorboard --logdir runs/train --port 6006 --bind_all

# 在本地设置 SSH 隧道
ssh -L 6006:localhost:6006 user@gpu-server

# 本地浏览器访问
http://localhost:6006
```

### 实时日志

```bash
# 查看训练日志
tail -f runs/train/obstacle_v8n_16gpu_*/results.csv
```

### GPU 使用率

```bash
# 监控所有 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat（更简洁）
pip install gpustat
gpustat -i 1
```

## 🕐 训练时间估算

```bash
python src/training/train_distributed.py estimate \
    --dataset-size 1000 \
    --epochs 100 \
    --gpus 16
```

示例输出：
```
==================================================
训练时间估算
==================================================
数据集大小: 1000 张
训练轮数: 100
GPU数量: 16
Batch大小: 256
--------------------------------------------------
GPU效率: 85%
有效加速: 13.6x
每轮时间: 0.4 分钟
总时间: 0.6 小时
==================================================
```

## 🔧 故障排除

### NCCL 错误

```bash
# 设置 NCCL 环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
```

### OOM (显存不足)

```bash
# 减少 batch size
python src/training/train_distributed.py ddp \
    --batch 128 \  # 减半
    --imgsz 480    # 或减小图像
```

### 多卡同步问题

```bash
# 使用 GLOO 后端替代 NCCL
export TORCH_DISTRIBUTED_BACKEND=gloo
```

## 📦 训练完成后

### 1. 下载模型

```bash
# 在本地执行
rsync -avz user@gpu-server:/path/to/project/llm/runs/train/*/weights/best.pt \
    ./models/
```

### 2. 继续 Day 3 优化

```bash
# 使用训练好的模型
python src/optimization/model_optimization.py onnx models/best.pt
python src/optimization/model_optimization.py tensorrt models/best.pt
```

---

## 📋 命令速查表

| 任务 | 命令 |
|------|------|
| 检查环境 | `bash scripts/train_multi_gpu.sh check` |
| 开始训练 | `bash scripts/train_multi_gpu.sh train 100 all` |
| 指定GPU | `bash scripts/train_multi_gpu.sh train 100 0,1,2,3,4,5,6,7` |
| 性能测试 | `bash scripts/train_multi_gpu.sh benchmark` |
| 时间估算 | `bash scripts/train_multi_gpu.sh estimate 1000 100` |
| SLURM脚本 | `bash scripts/train_multi_gpu.sh slurm` |

