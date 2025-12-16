#!/bin/bash
"""
半自动标注脚本
使用 YOLO11n 预训练模型对新数据进行预标注，加速标注过程
"""

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "半自动标注工具"
echo "============================================================"

# 参数
INPUT_DIR=${1:-"data/seed_dataset_v2"}
OUTPUT_DIR=${2:-"$INPUT_DIR/auto_labels"}
MODEL=${3:-"yolo11n.pt"}
CONF_THRESHOLD=${4:-0.25}

echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "模型文件: $MODEL"
echo "置信度阈值: $CONF_THRESHOLD"
echo ""

# 检查输入
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo -e "${RED}错误: 模型文件不存在: $MODEL${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行预标注
echo "开始预标注..."
python scripts/auto_annotate.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL" \
    --conf "$CONF_THRESHOLD" \
    --save-txt \
    --save-conf

echo ""
echo "============================================================"
echo -e "${GREEN}✓ 预标注完成！${NC}"
echo "============================================================"
echo ""
echo "输出位置: $OUTPUT_DIR"
echo ""
echo "下一步:"
echo "  1. 使用 LabelImg 检查和修正预标注:"
echo "     labelImg $INPUT_DIR $OUTPUT_DIR"
echo ""
echo "  2. 或导入到 Label Studio 进行团队审核"
echo ""
