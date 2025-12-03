#!/bin/bash
# 编译和运行C++版本的.bag提取工具

set -e

echo "=========================================="
echo "编译 RealSense .bag 提取工具"
echo "=========================================="

# 检查依赖
if ! pkg-config --exists realsense2; then
    echo "错误: librealsense2 未安装"
    echo ""
    echo "macOS安装方法:"
    echo "  brew install librealsense"
    echo ""
    exit 1
fi

if ! pkg-config --exists opencv4; then
    echo "错误: OpenCV 未安装"
    echo ""
    echo "macOS安装方法:"
    echo "  brew install opencv"
    echo ""
    exit 1
fi

# 编译
echo "正在编译..."
g++ -std=c++11 \
    src/data_processing/extract_bag_frames.cpp \
    $(pkg-config --cflags --libs realsense2 opencv4) \
    -o extract_bag_frames

echo "✓ 编译成功！"
echo ""
echo "=========================================="
echo "开始提取帧..."
echo "=========================================="

# 运行（处理所有.bag文件）
for bag_file in data/raw/*.bag; do
    if [ -f "$bag_file" ]; then
        echo "处理: $bag_file"
        ./extract_bag_frames "$bag_file" "data/frames" 5
        echo ""
    fi
done

echo "=========================================="
echo "✓ 所有文件处理完成！"
echo "=========================================="
