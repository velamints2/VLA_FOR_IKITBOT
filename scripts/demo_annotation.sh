#!/bin/bash
"""
标注工具演示脚本 - 展示完整工作流
"""

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo -e "${BLUE}半自动标注工具演示${NC}"
echo "============================================================"
echo ""
echo "本演示将展示如何使用标注工具对 seed_dataset_v2 进行标注"
echo ""

# 检查数据集
if [ ! -d "data/seed_dataset_v2" ]; then
    echo -e "${RED}错误: seed_dataset_v2 不存在${NC}"
    echo "请先运行数据提取脚本"
    exit 1
fi

IMAGE_COUNT=$(find data/seed_dataset_v2 -name "*.jpg" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ 找到 $IMAGE_COUNT 张图像${NC}"
echo ""

# 选项菜单
echo "请选择演示模式:"
echo ""
echo "  ${BLUE}1) 完整演示${NC} - 预标注 + LabelImg 检查 (推荐)"
echo "  ${BLUE}2) 仅预标注${NC} - 生成自动标注但不打开 LabelImg"
echo "  ${BLUE}3) Label Studio${NC} - 启动 Web 标注平台"
echo "  ${BLUE}4) 查看文档${NC} - 显示使用说明"
echo ""
read -p "请输入选项 (1-4, 默认=1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "============================================================"
        echo -e "${BLUE}模式 1: 完整演示${NC}"
        echo "============================================================"
        echo ""
        echo "步骤 1/3: 运行预标注 (使用 YOLO11n)"
        echo ""
        
        python scripts/auto_annotate.py \
            --input data/seed_dataset_v2 \
            --output data/seed_dataset_v2/demo_labels \
            --model yolo11n.pt \
            --conf 0.25 \
            --visualize
        
        echo ""
        echo "步骤 2/3: 查看可视化结果"
        echo ""
        
        if [ -d "data/seed_dataset_v2/demo_labels/visualizations" ]; then
            FIRST_VIS=$(ls data/seed_dataset_v2/demo_labels/visualizations/*.jpg | head -1)
            if [ -f "$FIRST_VIS" ]; then
                echo -e "${GREEN}可视化结果已保存到: data/seed_dataset_v2/demo_labels/visualizations/${NC}"
                echo ""
                read -p "按 Enter 查看第一张可视化图像..."
                open "$FIRST_VIS" || echo "请手动打开: $FIRST_VIS"
            fi
        fi
        
        echo ""
        echo "步骤 3/3: 启动 LabelImg 检查和修正"
        echo ""
        echo "即将启动 LabelImg..."
        echo ""
        echo -e "${YELLOW}LabelImg 快捷键提示:${NC}"
        echo "  W - 创建矩形框"
        echo "  D - 下一张"
        echo "  A - 上一张"
        echo "  Ctrl+S - 保存"
        echo "  Del - 删除框"
        echo ""
        read -p "按 Enter 启动 LabelImg (或 Ctrl+C 退出)..."
        
        if command -v labelImg &> /dev/null; then
            labelImg data/seed_dataset_v2 data/seed_dataset_v2/demo_labels
        else
            echo -e "${RED}LabelImg 未安装${NC}"
            echo "运行: pip install labelImg"
        fi
        ;;
        
    2)
        echo ""
        echo "============================================================"
        echo -e "${BLUE}模式 2: 仅预标注${NC}"
        echo "============================================================"
        echo ""
        
        python scripts/auto_annotate.py \
            --input data/seed_dataset_v2 \
            --output data/seed_dataset_v2/demo_labels \
            --model yolo11n.pt \
            --conf 0.25 \
            --visualize
        
        echo ""
        echo -e "${GREEN}✓ 预标注完成！${NC}"
        echo ""
        echo "输出位置:"
        echo "  - 标注文件: data/seed_dataset_v2/demo_labels/*.txt"
        echo "  - 可视化: data/seed_dataset_v2/demo_labels/visualizations/"
        echo ""
        echo "下一步:"
        echo "  labelImg data/seed_dataset_v2 data/seed_dataset_v2/demo_labels"
        ;;
        
    3)
        echo ""
        echo "============================================================"
        echo -e "${BLUE}模式 3: Label Studio${NC}"
        echo "============================================================"
        echo ""
        
        if ! command -v label-studio &> /dev/null; then
            echo -e "${RED}Label Studio 未安装${NC}"
            echo ""
            read -p "是否现在安装? (y/n): " install_ls
            if [ "$install_ls" = "y" ] || [ "$install_ls" = "Y" ]; then
                pip install label-studio
            else
                echo "退出演示"
                exit 0
            fi
        fi
        
        echo "启动 Label Studio..."
        echo ""
        echo "完成后请在浏览器中:"
        echo "  1. 创建项目"
        echo "  2. 导入配置: label_studio/config.xml"
        echo "  3. 添加数据: data/seed_dataset_v2"
        echo ""
        
        bash scripts/start_label_studio.sh
        ;;
        
    4)
        echo ""
        echo "============================================================"
        echo -e "${BLUE}模式 4: 查看文档${NC}"
        echo "============================================================"
        echo ""
        
        echo "📚 可用文档:"
        echo ""
        echo "  1. 标注工具集成指南:"
        echo "     docs/annotation_tools_guide.md"
        echo ""
        echo "  2. Label Studio 完整教程:"
        echo "     label_studio/README.md"
        echo ""
        echo "  3. 项目 README:"
        echo "     README.md"
        echo ""
        
        read -p "按 Enter 打开集成指南..."
        
        if command -v code &> /dev/null; then
            code docs/annotation_tools_guide.md
        elif [ -f docs/annotation_tools_guide.md ]; then
            cat docs/annotation_tools_guide.md
        fi
        ;;
        
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo -e "${GREEN}演示完成！${NC}"
echo "============================================================"
echo ""
echo "📊 效率对比:"
echo "  - 纯手工标注: 200张 ≈ 3-4 小时"
echo "  - 预标注+修正: 200张 ≈ 1 小时 (节省 70%)"
echo ""
echo "🚀 开始标注:"
echo "  bash scripts/auto_annotate.sh data/seed_dataset_v2"
echo "  labelImg data/seed_dataset_v2 data/seed_dataset_v2/auto_labels"
echo ""
