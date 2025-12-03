# Jetson Nano 部署环境配置指南

## 概述
本文档描述如何在Jetson Nano上配置障碍物检测系统的部署环境。

## 硬件要求
- Jetson Nano 开发板
- microSD卡（至少32GB，推荐64GB）
- 电源适配器（5V 4A）
- RGBD摄像头（如RealSense D435）

## 1. 系统安装

### 1.1 烧录JetPack
1. 下载JetPack 4.6或更高版本
2. 使用Balena Etcher烧录到SD卡
3. 首次启动配置用户名密码

### 1.2 系统更新
```bash
sudo apt update
sudo apt upgrade -y
```

## 2. 安装CUDA和TensorRT

JetPack已预装CUDA和TensorRT，验证安装：
```bash
# 检查CUDA
nvcc --version

# 检查TensorRT
dpkg -l | grep TensorRT
```

## 3. 安装Python环境

### 3.1 安装pip
```bash
sudo apt install python3-pip -y
pip3 install --upgrade pip
```

### 3.2 安装基础依赖
```bash
# NumPy和OpenCV
sudo apt install python3-numpy python3-opencv -y

# 或从pip安装（可能需要从源编译）
pip3 install numpy
```

## 4. 安装推理框架

### 4.1 安装NVIDIA jetson-inference
```bash
# 克隆仓库
cd ~
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference

# 编译和安装
mkdir build
cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 4.2 安装ONNX Runtime
```bash
# 从NVIDIA预编译版本安装
wget https://nvidia.box.com/shared/static/xxx.whl
pip3 install onnxruntime_gpu-1.x.x-cp36-cp36m-linux_aarch64.whl
```

## 5. 安装RealSense SDK

### 5.1 安装依赖
```bash
sudo apt install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev -y
sudo apt install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev -y
```

### 5.2 编译安装librealsense
```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build

cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true -DBUILD_PYTHON_BINDINGS=true
make -j$(nproc)
sudo make install
```

### 5.3 配置udev规则
```bash
cd ~/librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 5.4 验证安装
```bash
# 测试RealSense
realsense-viewer

# 测试Python绑定
python3 -c "import pyrealsense2 as rs; print('RealSense OK')"
```

## 6. 测试环境

### 6.1 测试RGBD摄像头
```bash
cd /path/to/llm
python3 scripts/test_rgbd_stream.py
```

### 6.2 测试TensorRT推理
```bash
# 使用示例模型测试
cd ~/jetson-inference/build/aarch64/bin
./imagenet-console test.jpg output.jpg
```

## 7. 性能优化

### 7.1 设置最大性能模式
```bash
# 查看当前模式
sudo nvpmodel -q

# 设置为最大性能模式（MODE 0）
sudo nvpmodel -m 0

# 设置风扇为最大速度
sudo jetson_clocks
```

### 7.2 增加swap空间
```bash
# 创建8GB swap文件
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 8. 基准测试

运行基准测试脚本：
```bash
python3 scripts/benchmark_jetson.py
```

预期性能指标：
- YOLOv10n (FP16): 15-20 FPS @ 640x640
- YOLOv10n (INT8): 25-30 FPS @ 640x640
- 内存占用: < 2GB

## 9. 常见问题

### Q1: CUDA out of memory
**解决方案**：
- 减小输入图像分辨率
- 使用INT8量化
- 增加swap空间

### Q2: RealSense无法识别
**解决方案**：
```bash
# 重新加载USB设备
sudo udevadm control --reload-rules
sudo udevadm trigger

# 检查USB连接
lsusb | grep Intel
```

### Q3: TensorRT转换失败
**解决方案**：
- 确保ONNX opset版本兼容
- 检查TensorRT版本（建议8.x）
- 简化模型结构

## 10. 下一步

环境配置完成后：
1. 部署训练好的模型（参考 `src/deployment/deploy_jetson.py`）
2. 运行实时检测测试
3. 集成到机器人控制系统

## 参考资料
- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [RealSense for Jetson](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
