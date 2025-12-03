# Jetson Nano 部署环境配置指南

## 概述

本文档描述如何在 Jetson Nano 上配置障碍物检测系统的部署环境。

**已验证配置**: 
- Jetson Nano Developer Kit
- JetPack 4.6.x (L4T R32.7.1)
- CUDA 10.2 + cuDNN 8.2.1 + TensorRT 8.2.1

## 硬件要求

| 组件 | 规格 |
|------|------|
| 开发板 | Jetson Nano 4GB |
| microSD | ≥32GB (推荐 64GB) |
| 电源 | 5V 4A DC 电源适配器 |
| 摄像头 | CSI (IMX219) 或 USB 摄像头 |

## 1. 快速配置 (推荐)

上传并运行一键配置脚本：

```bash
# 从开发机上传脚本
scp scripts/setup_jetson.sh user@<jetson-ip>:~/

# SSH 到 Jetson
ssh user@<jetson-ip>

# 运行配置脚本
bash ~/setup_jetson.sh
```

## 2. 手动配置

### 2.1 安装 pip3

```bash
sudo apt update
sudo apt install -y python3-pip
```

### 2.2 配置 pip 镜像 (加速下载)

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
EOF
```

### 2.3 安装 PyTorch for Jetson

```bash
# 下载 PyTorch 1.10.0 (JetPack 4.6)
cd /tmp
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
    -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 安装依赖
sudo apt install -y libopenblas-base libopenmpi-dev libomp-dev

# 安装 PyTorch
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

### 2.4 安装其他依赖

```bash
pip3 install pillow pyyaml tqdm
```

## 3. 验证安装

```bash
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

import tensorrt as trt
print('TensorRT:', trt.__version__)

import cv2
print('OpenCV:', cv2.__version__)
"
```

预期输出：
```
PyTorch: 1.10.0
CUDA: True
GPU: NVIDIA Tegra X1
TensorRT: 8.2.1.8
OpenCV: 4.1.1
```

## 4. 性能优化

### 4.1 设置最大性能模式

```bash
# 设置 MAXN 模式
sudo nvpmodel -m 0

# 启用 jetson_clocks
sudo jetson_clocks
```

### 4.2 增加 Swap 空间

```bash
# 创建 4GB swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 5. 测试摄像头

```bash
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print('Camera:', frame.shape[1], 'x', frame.shape[0])
else:
    print('Camera: Failed')
cap.release()
"
```

## 6. 运行环境测试

```bash
# 上传测试脚本
scp src/deployment/jetson_test.py user@<jetson-ip>:~/

# 运行完整测试
python3 ~/jetson_test.py all
```

## 7. 当前环境状态 (已验证)

| 组件 | 版本 | 状态 |
|------|------|------|
| L4T | R32.7.1 | ✅ |
| CUDA | 10.2.300 | ✅ |
| cuDNN | 8.2.1.32 | ✅ |
| TensorRT | 8.2.1.8 | ✅ |
| Python | 3.6.9 | ✅ |
| PyTorch | 1.10.0 | ✅ |
| OpenCV | 4.1.1 | ✅ |
| Pillow | 8.4.0 | ✅ |
| pip3 | 21.3.1 | ✅ |

## 8. SSH 连接信息

```bash
# 当前测试设备
ssh velamints@192.168.0.219
```

## 参考资料

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
