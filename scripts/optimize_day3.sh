#!/bin/bash
# Day 3: 模型轻量化脚本
# 使用方法: bash scripts/optimize_day3.sh [command] [options]

set -e

# 激活环境
source /Users/macbookair/opt/anaconda3/bin/activate obstacle_detection

# 切换到项目目录
cd /Users/macbookair/Documents/trae_projects/llm

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 参数
CMD=${1:-"help"}
MODEL=${2:-"runs/train/obstacle_detection/weights/best.pt"}

echo "=========================================="
echo "Day 3: 模型轻量化"
echo "=========================================="

case $CMD in
    check)
        echo "检查依赖..."
        python src/optimization/model_optimization.py check
        ;;
    
    info)
        echo "查看模型信息: $MODEL"
        python src/optimization/model_optimization.py info "$MODEL"
        ;;
    
    onnx)
        echo "导出 ONNX 模型: $MODEL"
        python src/optimization/model_optimization.py onnx "$MODEL" --imgsz 640
        ;;
    
    tensorrt)
        echo "导出 TensorRT 引擎: $MODEL"
        python src/optimization/model_optimization.py tensorrt "$MODEL" --half
        ;;
    
    benchmark)
        DEVICE=${3:-"cpu"}
        echo "性能测试: $MODEL (设备: $DEVICE)"
        python src/optimization/model_optimization.py benchmark "$MODEL" --device "$DEVICE"
        ;;
    
    compare)
        OPTIMIZED=${3:-""}
        if [ -z "$OPTIMIZED" ]; then
            echo "用法: bash scripts/optimize_day3.sh compare <原始模型> <优化模型>"
            exit 1
        fi
        echo "对比模型..."
        python src/optimization/model_optimization.py compare "$MODEL" "$OPTIMIZED"
        ;;
    
    deploy)
        OUTPUT=${3:-"deployment"}
        echo "创建部署包: $MODEL -> $OUTPUT"
        python src/optimization/model_optimization.py deploy "$MODEL" --output "$OUTPUT"
        ;;
    
    all)
        # 完整流程
        echo "执行完整优化流程..."
        
        # 1. 检查依赖
        python src/optimization/model_optimization.py check
        
        # 2. 查看原始模型信息
        python src/optimization/model_optimization.py info "$MODEL"
        
        # 3. 原始模型基准测试
        echo ""
        echo "原始模型性能测试..."
        python src/optimization/model_optimization.py benchmark "$MODEL" --device cpu --runs 50
        
        # 4. 导出 ONNX
        echo ""
        echo "导出 ONNX..."
        python src/optimization/model_optimization.py onnx "$MODEL"
        
        # 5. 创建部署包
        echo ""
        echo "创建部署包..."
        python src/optimization/model_optimization.py deploy "$MODEL" --output deployment
        
        echo ""
        echo "=========================================="
        echo "✓ 优化流程完成！"
        echo "=========================================="
        ;;
    
    help|*)
        echo "使用方法:"
        echo "  bash scripts/optimize_day3.sh check                    # 检查依赖"
        echo "  bash scripts/optimize_day3.sh info <model>             # 查看模型信息"
        echo "  bash scripts/optimize_day3.sh onnx <model>             # 导出 ONNX"
        echo "  bash scripts/optimize_day3.sh tensorrt <model>         # 导出 TensorRT"
        echo "  bash scripts/optimize_day3.sh benchmark <model> [device] # 性能测试"
        echo "  bash scripts/optimize_day3.sh compare <orig> <opt>     # 模型对比"
        echo "  bash scripts/optimize_day3.sh deploy <model> [output]  # 创建部署包"
        echo "  bash scripts/optimize_day3.sh all <model>              # 完整流程"
        ;;
esac
