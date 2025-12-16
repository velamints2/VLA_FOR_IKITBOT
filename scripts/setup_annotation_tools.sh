#!/bin/bash
"""
标注工具安装和配置脚本
支持 LabelImg (快速标注) 和 Label Studio (协作标注)
"""

set -e

echo "============================================================"
echo "标注工具安装向导"
echo "============================================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 conda 环境
check_conda_env() {
    if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
        echo -e "${RED}错误: 请先激活 obstacle_detection 环境${NC}"
        echo "运行: conda activate obstacle_detection"
        exit 1
    fi
    echo -e "${GREEN}✓ Conda 环境: $CONDA_DEFAULT_ENV${NC}"
}

# 安装 LabelImg
install_labelimg() {
    echo ""
    echo "============================================================"
    echo "安装 LabelImg (桌面快速标注工具)"
    echo "============================================================"
    
    if command -v labelImg &> /dev/null; then
        echo -e "${GREEN}✓ LabelImg 已安装${NC}"
        labelImg --version || echo "LabelImg 版本信息不可用"
    else
        echo "正在安装 LabelImg..."
        pip install labelImg
        echo -e "${GREEN}✓ LabelImg 安装完成${NC}"
    fi
    
    echo ""
    echo "使用方法:"
    echo "  labelImg data/seed_dataset_v2 data/seed_dataset_v2/classes.txt"
    echo ""
    echo "快捷键:"
    echo "  W - 创建矩形框"
    echo "  D - 下一张图片"
    echo "  A - 上一张图片"
    echo "  Ctrl+S - 保存"
    echo "  Del - 删除选中的框"
}

# 安装 Label Studio
install_label_studio() {
    echo ""
    echo "============================================================"
    echo "安装 Label Studio (Web协作标注平台)"
    echo "============================================================"
    
    if command -v label-studio &> /dev/null; then
        echo -e "${GREEN}✓ Label Studio 已安装${NC}"
        label-studio --version
    else
        echo "正在安装 Label Studio..."
        pip install label-studio
        echo -e "${GREEN}✓ Label Studio 安装完成${NC}"
    fi
    
    echo ""
    echo "使用方法:"
    echo "  1. 启动服务: label-studio start"
    echo "  2. 浏览器打开: http://localhost:8080"
    echo "  3. 创建项目并导入数据"
}

# 安装 Label Studio ML Backend (用于半自动标注)
install_label_studio_ml() {
    echo ""
    echo "============================================================"
    echo "安装 Label Studio ML Backend (半自动标注)"
    echo "============================================================"
    
    if python -c "import label_studio_ml" 2>/dev/null; then
        echo -e "${GREEN}✓ Label Studio ML Backend 已安装${NC}"
    else
        echo "正在安装 Label Studio ML Backend..."
        pip install label-studio-ml
        echo -e "${GREEN}✓ Label Studio ML Backend 安装完成${NC}"
    fi
}

# 创建 Label Studio 配置
create_label_studio_config() {
    echo ""
    echo "============================================================"
    echo "创建 Label Studio 配置文件"
    echo "============================================================"
    
    mkdir -p label_studio
    
    cat > label_studio/config.xml << 'EOF'
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="wire" background="red"/>
    <Label value="slipper" background="blue"/>
    <Label value="sock" background="green"/>
    <Label value="cable" background="yellow"/>
    <Label value="toy" background="purple"/>
    <Label value="obstacle" background="orange"/>
  </RectangleLabels>
</View>
EOF
    
    echo -e "${GREEN}✓ 配置文件已创建: label_studio/config.xml${NC}"
}

# 主流程
main() {
    echo "开始安装标注工具..."
    echo ""
    
    check_conda_env
    
    echo ""
    echo "选择要安装的工具:"
    echo "  1) LabelImg (推荐：快速标注)"
    echo "  2) Label Studio (推荐：团队协作)"
    echo "  3) 全部安装"
    echo "  4) 仅创建配置文件"
    echo ""
    read -p "请输入选项 (1-4, 默认=3): " choice
    choice=${choice:-3}
    
    case $choice in
        1)
            install_labelimg
            ;;
        2)
            install_label_studio
            install_label_studio_ml
            create_label_studio_config
            ;;
        3)
            install_labelimg
            install_label_studio
            install_label_studio_ml
            create_label_studio_config
            ;;
        4)
            create_label_studio_config
            ;;
        *)
            echo -e "${RED}无效选项${NC}"
            exit 1
            ;;
    esac
    
    echo ""
    echo "============================================================"
    echo -e "${GREEN}✓ 安装完成！${NC}"
    echo "============================================================"
    echo ""
    echo "下一步:"
    echo "  1. 使用 LabelImg 快速标注:"
    echo "     labelImg data/seed_dataset_v2 configs/classes.txt"
    echo ""
    echo "  2. 使用 Label Studio (需要先运行半自动预标注):"
    echo "     bash scripts/auto_annotate.sh"
    echo "     label-studio start"
    echo ""
}

main
