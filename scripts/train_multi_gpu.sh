#!/bin/bash
#
# Day 2: 16x RTX 2080 多GPU训练脚本
# 
# 使用方法:
#   ./scripts/train_multi_gpu.sh [command] [options]
#
# 命令:
#   check       - 检查GPU环境
#   train       - 开始训练
#   benchmark   - 性能测试
#   estimate    - 估算时间
#

set -e

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 默认参数
MODEL="yolo11n.pt"
EPOCHS=100
IMGSZ=640
GPUS="all"
DATA="configs/data.yaml"
WORKERS=8
AMP="--amp"
CACHE="--cache"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${GREEN}"
    echo "============================================================"
    echo "  16x RTX 2080 多GPU训练"
    echo "============================================================"
    echo -e "${NC}"
}

print_usage() {
    echo "使用方法: $0 [command] [options]"
    echo ""
    echo "命令:"
    echo "  check                    检查GPU环境"
    echo "  train [epochs] [gpus]    开始训练 (默认: 100 epochs, all gpus)"
    echo "  benchmark                多GPU性能测试"
    echo "  estimate [size] [epochs] 估算训练时间"
    echo "  slurm                    生成SLURM脚本"
    echo ""
    echo "示例:"
    echo "  $0 check"
    echo "  $0 train 100 all"
    echo "  $0 train 50 0,1,2,3,4,5,6,7"
    echo "  $0 estimate 1000 100"
}

check_conda() {
    if command -v conda &> /dev/null; then
        echo -e "${GREEN}✓ Conda 已安装${NC}"
        # 激活环境
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate obstacle_detection 2>/dev/null || {
            echo -e "${YELLOW}⚠ 创建conda环境...${NC}"
            conda create -n obstacle_detection python=3.8 -y
            conda activate obstacle_detection
        }
        echo -e "${GREEN}✓ Conda环境: obstacle_detection${NC}"
    else
        echo -e "${RED}✗ Conda 未安装${NC}"
        exit 1
    fi
}

install_deps() {
    echo -e "${YELLOW}检查依赖...${NC}"
    
    # PyTorch with CUDA
    python -c "import torch" 2>/dev/null || {
        echo -e "${YELLOW}安装 PyTorch (CUDA 11.8)...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
    
    # Ultralytics
    python -c "from ultralytics import YOLO" 2>/dev/null || {
        echo -e "${YELLOW}安装 Ultralytics...${NC}"
        pip install ultralytics
    }
    
    echo -e "${GREEN}✓ 依赖检查完成${NC}"
}

run_check() {
    print_header
    check_conda
    install_deps
    echo ""
    python src/training/train_distributed.py check
}

run_train() {
    local epochs=${1:-$EPOCHS}
    local gpus=${2:-$GPUS}
    
    print_header
    check_conda
    install_deps
    
    echo ""
    echo -e "${GREEN}开始训练:${NC}"
    echo "  模型: $MODEL"
    echo "  轮数: $epochs"
    echo "  GPU: $gpus"
    echo "  数据: $DATA"
    echo ""
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$(echo $gpus | tr ' ' ',')
    export OMP_NUM_THREADS=$WORKERS
    export NCCL_DEBUG=WARN
    
    # 大batch训练推荐设置
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=eth0
    
    python src/training/train_distributed.py ddp \
        --data "$DATA" \
        --model "$MODEL" \
        --epochs "$epochs" \
        --imgsz "$IMGSZ" \
        --gpus "$gpus" \
        --workers "$WORKERS" \
        $AMP $CACHE
}

run_benchmark() {
    print_header
    check_conda
    install_deps
    python src/training/train_distributed.py benchmark --gpus all
}

run_estimate() {
    local size=${1:-1000}
    local epochs=${2:-100}
    
    print_header
    python src/training/train_distributed.py estimate \
        --dataset-size "$size" \
        --epochs "$epochs" \
        --gpus 16
}

run_slurm() {
    print_header
    python src/training/train_distributed.py slurm --gpus 16 --epochs 100
}

# 主程序
case "${1:-help}" in
    check)
        run_check
        ;;
    train)
        run_train "${2:-100}" "${3:-all}"
        ;;
    benchmark)
        run_benchmark
        ;;
    estimate)
        run_estimate "${2:-1000}" "${3:-100}"
        ;;
    slurm)
        run_slurm
        ;;
    *)
        print_usage
        ;;
esac
