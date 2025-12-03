#!/bin/bash
# Day 2: 基线模型训练启动脚本
# 使用方法: bash scripts/train_day2.sh [validate|train]

set -e

# 激活环境
source /Users/macbookair/opt/anaconda3/bin/activate obstacle_detection

# 切换到项目目录
cd /Users/macbookair/Documents/trae_projects/llm

# 设置环境变量（解决MPS兼容性问题）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 参数
MODE=${1:-"train"}
MODEL=${2:-"yolov8n.pt"}
EPOCHS=${3:-50}

echo "=========================================="
echo "Day 2: 基线模型训练"
echo "=========================================="
echo "模式: $MODE"
echo "模型: $MODEL"
echo "轮数: $EPOCHS"
echo "=========================================="

if [ "$MODE" == "validate" ]; then
    echo "运行快速验证..."
    python src/training/train_baseline.py validate \
        --data configs/data.yaml \
        --model $MODEL

elif [ "$MODE" == "train" ]; then
    echo "开始正式训练..."
    python src/training/train_baseline.py train \
        --data configs/data.yaml \
        --model $MODEL \
        --epochs $EPOCHS \
        --imgsz 640 \
        --batch 8 \
        --device auto

elif [ "$MODE" == "eval" ]; then
    # 评估最新模型
    LATEST_MODEL=$(ls -t runs/train/*/weights/best.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_MODEL" ]; then
        echo "评估模型: $LATEST_MODEL"
        python src/training/train_baseline.py eval "$LATEST_MODEL" --data configs/data.yaml
    else
        echo "错误: 未找到训练好的模型"
        exit 1
    fi

else
    echo "未知模式: $MODE"
    echo "使用方法: bash scripts/train_day2.sh [validate|train|eval] [model] [epochs]"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 完成！"
echo "=========================================="
