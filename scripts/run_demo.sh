#!/bin/bash
#
# ROS Demo 快速启动脚本
# 扫地机器人障碍物检测系统
#
# 使用方法:
#   ./scripts/run_demo.sh camera     # 使用摄像头实时检测
#   ./scripts/run_demo.sh video path/to/video.mp4  # 处理视频文件
#   ./scripts/run_demo.sh ros        # 启动完整ROS Demo
#   ./scripts/run_demo.sh help       # 显示帮助
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo ""
    echo "=========================================="
    echo "  扫地机器人障碍物检测系统 - Demo启动脚本"
    echo "=========================================="
    echo ""
    echo "用法: $0 <command> [options]"
    echo ""
    echo "命令:"
    echo "  camera [device]       使用摄像头实时检测 (独立模式)"
    echo "  video <path> [output] 处理视频文件"
    echo "  batch <dir> [outdir]  批量处理视频目录"
    echo "  ros                   启动ROS Demo (需要ROS环境)"
    echo "  test                  运行快速测试"
    echo "  help                  显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 camera             # 使用默认摄像头 (/dev/video0)"
    echo "  $0 camera 1           # 使用 /dev/video1"
    echo "  $0 video demo.mp4     # 处理视频文件"
    echo "  $0 video demo.mp4 out.mp4  # 指定输出文件"
    echo "  $0 batch videos/ results/  # 批量处理"
    echo "  $0 ros                # 启动ROS Demo"
    echo ""
    echo "环境变量:"
    echo "  MODEL_PATH    模型路径 (默认: models/best.pt)"
    echo "  CONF_THRESH   置信度阈值 (默认: 0.5)"
    echo "  DEVICE        推理设备 (默认: cpu)"
    echo ""
}

# 检查Python依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "未找到 python3"
        exit 1
    fi
    
    # 检查关键包
    python3 -c "import ultralytics" 2>/dev/null || {
        print_error "未安装 ultralytics，请运行: pip install ultralytics"
        exit 1
    }
    
    python3 -c "import cv2" 2>/dev/null || {
        print_error "未安装 opencv-python，请运行: pip install opencv-python"
        exit 1
    }
    
    # 检查模型文件
    MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/best.pt}"
    if [ ! -f "$MODEL_PATH" ]; then
        print_error "模型文件不存在: $MODEL_PATH"
        print_info "请确保已将训练好的模型放在 models/ 目录下"
        exit 1
    fi
    
    print_success "依赖检查通过"
}

# 摄像头模式
run_camera() {
    local device="${1:-0}"
    
    check_dependencies
    
    print_info "启动摄像头实时检测..."
    print_info "摄像头: $device"
    print_info "模型: ${MODEL_PATH:-models/best.pt}"
    print_info "按 'q' 退出, 按 's' 保存截图"
    echo ""
    
    cd "$PROJECT_ROOT"
    python3 src/ros_demo/obstacle_detector_node.py \
        --standalone \
        --source "$device" \
        --model "${MODEL_PATH:-models/best.pt}" \
        --conf "${CONF_THRESH:-0.5}" \
        --device "${DEVICE:-cpu}"
}

# 视频处理模式
run_video() {
    local input_path="$1"
    local output_path="$2"
    
    if [ -z "$input_path" ]; then
        print_error "请指定输入视频路径"
        show_help
        exit 1
    fi
    
    check_dependencies
    
    print_info "处理视频文件: $input_path"
    
    cd "$PROJECT_ROOT"
    
    if [ -n "$output_path" ]; then
        python3 src/ros_demo/video_processor_node.py \
            --input "$input_path" \
            --output "$output_path" \
            --model "${MODEL_PATH:-models/best.pt}" \
            --conf "${CONF_THRESH:-0.5}" \
            --device "${DEVICE:-cpu}"
    else
        python3 src/ros_demo/video_processor_node.py \
            --input "$input_path" \
            --model "${MODEL_PATH:-models/best.pt}" \
            --conf "${CONF_THRESH:-0.5}" \
            --device "${DEVICE:-cpu}" \
            --stats-only
    fi
}

# 批量处理模式
run_batch() {
    local input_dir="$1"
    local output_dir="${2:-$1/annotated}"
    
    if [ -z "$input_dir" ]; then
        print_error "请指定输入目录"
        show_help
        exit 1
    fi
    
    check_dependencies
    
    print_info "批量处理目录: $input_dir"
    print_info "输出目录: $output_dir"
    
    cd "$PROJECT_ROOT"
    python3 src/ros_demo/video_processor_node.py \
        --input "$input_dir" \
        --output "$output_dir" \
        --model "${MODEL_PATH:-models/best.pt}" \
        --conf "${CONF_THRESH:-0.5}" \
        --device "${DEVICE:-cpu}" \
        --report "$output_dir/report.md"
}

# ROS模式
run_ros() {
    print_info "启动ROS Demo..."
    
    # 检查ROS
    if [ -z "$ROS_DISTRO" ]; then
        print_error "ROS 环境未初始化"
        print_info "请先运行: source /opt/ros/<distro>/setup.bash"
        exit 1
    fi
    
    print_info "ROS 版本: $ROS_DISTRO"
    
    # 检查是否已编译
    if ! rospack find obstacle_detection &> /dev/null; then
        print_warning "obstacle_detection 包未找到"
        print_info "请先编译 ROS 包:"
        echo "  cd ~/catkin_ws && catkin_make"
        echo "  source devel/setup.bash"
        exit 1
    fi
    
    # 启动
    print_info "启动完整 Demo..."
    roslaunch obstacle_detection demo.launch realtime:=true
}

# 快速测试
run_test() {
    check_dependencies
    
    print_info "运行快速测试..."
    
    cd "$PROJECT_ROOT"
    
    # 测试模型加载
    print_info "测试模型加载..."
    python3 -c "
from ultralytics import YOLO
import os

model_path = '${MODEL_PATH:-models/best.pt}'
if not os.path.exists(model_path):
    print(f'模型不存在: {model_path}')
    exit(1)

model = YOLO(model_path)
print(f'模型加载成功: {model.model_name}')
print(f'类别: {model.names}')
"
    
    if [ $? -eq 0 ]; then
        print_success "测试通过!"
    else
        print_error "测试失败"
        exit 1
    fi
}

# 主入口
main() {
    local command="${1:-help}"
    
    case "$command" in
        camera)
            run_camera "$2"
            ;;
        video)
            run_video "$2" "$3"
            ;;
        batch)
            run_batch "$2" "$3"
            ;;
        ros)
            run_ros
            ;;
        test)
            run_test
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

