#!/bin/bash
"""
启动 Label Studio 和 ML Backend
"""

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "Label Studio 启动脚本"
echo "============================================================"

# 检查安装
check_installation() {
    if ! command -v label-studio &> /dev/null; then
        echo -e "${RED}错误: Label Studio 未安装${NC}"
        echo "运行: bash scripts/setup_annotation_tools.sh"
        exit 1
    fi
    echo -e "${GREEN}✓ Label Studio 已安装${NC}"
}

# 启动 ML Backend (可选)
start_ml_backend() {
    echo ""
    read -p "是否启动 ML Backend 进行半自动标注? (y/n, 默认=n): " use_ml
    use_ml=${use_ml:-n}
    
    if [ "$use_ml" = "y" ] || [ "$use_ml" = "Y" ]; then
        echo ""
        echo "启动 ML Backend..."
        
        # 设置环境变量
        export MODEL_PATH=${MODEL_PATH:-"yolo11n.pt"}
        export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-"0.25"}
        export IOU_THRESHOLD=${IOU_THRESHOLD:-"0.45"}
        
        echo "模型路径: $MODEL_PATH"
        echo "置信度阈值: $CONFIDENCE_THRESHOLD"
        
        # 启动 ML Backend
        if command -v label-studio-ml &> /dev/null; then
            # 初始化 ML Backend
            if [ ! -d "label_studio/ml_backend" ]; then
                mkdir -p label_studio/ml_backend
                cd label_studio/ml_backend
                label-studio-ml init yolo_backend \
                    --script ../../scripts/label_studio_ml_backend.py
                cd ../..
            fi
            
            # 启动服务
            cd label_studio/ml_backend/yolo_backend
            label-studio-ml start . --port 9090 &
            ML_BACKEND_PID=$!
            cd ../../..
            
            echo -e "${GREEN}✓ ML Backend 已启动 (PID: $ML_BACKEND_PID)${NC}"
            echo "  URL: http://localhost:9090"
            echo ""
            echo "在 Label Studio 中连接 ML Backend:"
            echo "  Settings -> Machine Learning -> Add Model"
            echo "  URL: http://localhost:9090"
            echo ""
        else
            echo -e "${YELLOW}⚠ label-studio-ml 未安装，跳过 ML Backend${NC}"
        fi
    fi
}

# 启动 Label Studio
start_label_studio() {
    echo ""
    echo "启动 Label Studio..."
    
    # 设置数据目录
    export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)
    
    # 启动服务
    label-studio start \
        --port 8080 \
        --data-dir label_studio/data
}

# 主流程
main() {
    check_installation
    start_ml_backend
    start_label_studio
}

# 清理函数
cleanup() {
    echo ""
    echo "正在关闭服务..."
    if [ ! -z "$ML_BACKEND_PID" ]; then
        kill $ML_BACKEND_PID 2>/dev/null || true
        echo "✓ ML Backend 已关闭"
    fi
}

trap cleanup EXIT

main
