#!/bin/bash
#
# Jetson Nano 环境配置脚本
# 适用于: JetPack 4.6.x (L4T R32.7.1)
#

set -e

echo "=============================================="
echo "  Jetson Nano 环境配置脚本"
echo "=============================================="

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为 Jetson
check_jetson() {
    if [ ! -f /etc/nv_tegra_release ]; then
        print_error "此脚本仅适用于 Jetson 设备"
        exit 1
    fi
    print_status "检测到 Jetson 设备"
    cat /etc/nv_tegra_release
}

# 1. 安装 pip3
install_pip() {
    print_status "检查 pip3..."
    if command -v pip3 &> /dev/null; then
        print_status "pip3 已安装: $(pip3 --version)"
    else
        print_status "安装 pip3..."
        sudo apt update
        sudo apt install -y python3-pip
    fi
    
    # 升级 pip
    print_status "升级 pip..."
    pip3 install --upgrade pip
}

# 2. 安装 PyTorch
install_pytorch() {
    print_status "检查 PyTorch..."
    
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        print_status "PyTorch 已安装: $TORCH_VERSION"
    else
        print_status "安装 PyTorch 1.10 for Jetson..."
        
        cd /tmp
        
        # PyTorch 1.10.0 for JetPack 4.6 (Python 3.6)
        TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
        TORCH_FILE="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
        
        if [ ! -f "$TORCH_FILE" ]; then
            print_status "下载 PyTorch..."
            wget -q --show-progress "$TORCH_URL" -O "$TORCH_FILE"
        fi
        
        print_status "安装 PyTorch..."
        pip3 install "$TORCH_FILE"
    fi
}

# 3. 安装 torchvision
install_torchvision() {
    print_status "检查 torchvision..."
    
    if python3 -c "import torchvision" 2>/dev/null; then
        TV_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)")
        print_status "torchvision 已安装: $TV_VERSION"
    else
        print_status "安装 torchvision 依赖..."
        sudo apt install -y libopenblas-base libopenmpi-dev libomp-dev
        
        print_status "安装 torchvision..."
        pip3 install torchvision==0.11.0
    fi
}

# 4. 升级 NumPy
upgrade_numpy() {
    print_status "升级 NumPy..."
    
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0")
    
    if [[ "$NUMPY_VERSION" < "1.19" ]]; then
        pip3 install --upgrade "numpy<1.20"  # Python 3.6 兼容
    else
        print_status "NumPy 版本足够新: $NUMPY_VERSION"
    fi
}

# 5. 安装其他依赖
install_deps() {
    print_status "安装其他依赖..."
    pip3 install pillow pyyaml tqdm
}

# 6. 安装 ONNX Runtime (可选)
install_onnxruntime() {
    print_status "检查 ONNX Runtime..."
    
    if python3 -c "import onnxruntime" 2>/dev/null; then
        ORT_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
        print_status "ONNX Runtime 已安装: $ORT_VERSION"
    else
        print_warning "ONNX Runtime 需要从源码编译，跳过..."
        print_warning "如需安装，请参考: https://elinux.org/Jetson_Zoo"
    fi
}

# 7. 验证安装
verify_installation() {
    echo ""
    echo "=============================================="
    echo "  安装验证"
    echo "=============================================="
    
    # Python
    python3 --version
    
    # PyTorch + CUDA
    python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
"
    
    # TensorRT
    python3 -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
    
    # OpenCV
    python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
    
    # NumPy
    python3 -c "import numpy; print('NumPy:', numpy.__version__)"
    
    echo ""
    echo -e "${GREEN}=============================================="
    echo "  ✅ 环境配置完成!"
    echo "==============================================${NC}"
}

# 主函数
main() {
    check_jetson
    
    echo ""
    read -p "是否继续安装? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        echo "取消安装"
        exit 0
    fi
    
    install_pip
    install_pytorch
    install_torchvision
    upgrade_numpy
    install_deps
    install_onnxruntime
    verify_installation
}

# 运行
main "$@"
