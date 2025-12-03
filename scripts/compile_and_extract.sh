#!/bin/bash
# 编译和运行C++版本的.bag提取工具 (无OpenCV依赖版)

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

# 编译
echo "正在编译..."
REALSENSE_PREFIX=$(brew --prefix librealsense)
echo "RealSense Prefix: ${REALSENSE_PREFIX}"

g++ -std=c++11 \
    src/data_processing/extract_bag_frames.cpp \
    -I"${REALSENSE_PREFIX}/include" \
    -L"${REALSENSE_PREFIX}/lib" \
    -lrealsense2 \
    -o extract_bag_frames

echo "✓ 编译成功！"
echo ""
echo "=========================================="
echo "开始提取帧..."
echo "=========================================="

# 运行（处理所有.bag文件）
# 检查是否有bag文件
shopt -s nullglob
bag_files=(data/raw/*.bag)
if [ ${#bag_files[@]} -eq 0 ]; then
    echo "警告: data/raw/ 目录下没有找到 .bag 文件"
    exit 0
fi

for bag_file in "${bag_files[@]}"; do
    echo "处理: $bag_file"
    # 为每个bag文件创建一个子目录
    filename=$(basename -- "$bag_file")
    filename="${filename%.*}"
    output_dir="data/frames/$filename"
    
    ./extract_bag_frames "$bag_file" "$output_dir" 5
    
    echo "转换 Raw -> Image ($filename)..."
    python src/data_processing/convert_raw_to_png.py "$output_dir"
    
    echo ""
done

echo "=========================================="
echo "✓ 所有文件处理完成！"
echo "=========================================="
