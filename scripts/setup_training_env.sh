#!/bin/bash
# 服务器训练环境搭建脚本

set -e  # 遇到错误立即退出

echo "======================================================"
echo "扫地机器人障碍物检测 - 训练环境搭建"
echo "======================================================"

# 环境名称
ENV_NAME="obstacle_detection"

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 ${ENV_NAME} 已存在"
    read -p "是否删除并重新创建？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在删除旧环境..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "保留现有环境，跳过创建步骤"
        conda activate ${ENV_NAME}
        exit 0
    fi
fi

# 创建Python虚拟环境
echo ""
echo "步骤 1/3: 创建Python虚拟环境 (Python 3.8)..."
conda create -n ${ENV_NAME} python=3.8 -y

# 激活环境
echo ""
echo "步骤 2/3: 激活环境..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 安装依赖
echo ""
echo "步骤 3/3: 安装依赖包..."
echo "  - 安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "  - 安装YOLO和OpenCV..."
pip install ultralytics opencv-python

echo "  - 安装其他依赖..."
pip install numpy matplotlib pillow pyyaml tqdm pandas scikit-learn seaborn

echo "  - 安装ONNX工具..."
pip install onnx onnxruntime

echo "  - 安装RGBD处理库..."
pip install pyrealsense2 open3d

echo "  - 安装开发工具..."
pip install jupyter ipython

# 验证安装
echo ""
echo "======================================================"
echo "验证安装..."
echo "======================================================"

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics YOLO: OK')"

echo ""
echo "======================================================"
echo "✅ 环境搭建完成！"
echo "======================================================"
echo ""
echo "使用方法："
echo "  conda activate ${ENV_NAME}"
echo ""
echo "验证环境："
echo "  python scripts/check_environment.py"
echo ""
