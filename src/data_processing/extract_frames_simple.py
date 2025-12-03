"""
简化版帧提取工具 - 直接从bag文件提取（使用rosbag）
如果没有rosbag，则处理普通视频文件
"""
import os
import sys
import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("请运行: pip install opencv-python numpy")
    sys.exit(1)


def extract_frames_from_video(video_path, output_dir, sample_rate=2):
    """从普通视频文件提取帧"""
    print(f"正在处理视频: {video_path}")
    
    rgb_dir = Path(output_dir) / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {fps:.2f} FPS, 总帧数: {total_frames}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frame_path = rgb_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"已提取 {saved_count} 帧...")
        
        frame_count += 1
    
    cap.release()
    print(f"✓ 完成！共提取 {saved_count} 帧到 {rgb_dir}")
    return saved_count


def extract_from_bag_fallback(bag_path, output_dir, sample_rate=2):
    """
    .bag文件回退方案
    macOS上pyrealsense2难以安装，建议：
    1. 在Linux/Windows上用RealSense Viewer导出为视频
    2. 或使用rosbag工具提取
    """
    print(f"检测到.bag文件: {bag_path}")
    print("\n⚠️  pyrealsense2在macOS上难以安装")
    print("\n建议方案：")
    print("1. 使用RealSense Viewer将.bag导出为.mp4视频")
    print("   - 打开RealSense Viewer")
    print("   - 加载.bag文件")
    print("   - 导出为视频格式")
    print("\n2. 在Linux/Windows环境处理（有pyrealsense2）")
    print("\n3. 使用rosbag工具（如果安装了ROS）：")
    print(f"   rosbag play {bag_path}")
    print(f"   并录制为视频")
    
    print("\n临时解决方案：")
    print("请提供.mp4或.avi格式的视频文件")
    return 0


def main():
    parser = argparse.ArgumentParser(description="从视频提取帧")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("--output", "-o", default="data/frames", 
                       help="输出目录")
    parser.add_argument("--sample-rate", "-s", type=int, default=2,
                       help="采样率")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在 {args.input}")
        return 1
    
    file_ext = input_path.suffix.lower()
    
    if file_ext == ".bag":
        saved = extract_from_bag_fallback(args.input, args.output, args.sample_rate)
    else:
        saved = extract_frames_from_video(args.input, args.output, args.sample_rate)
    
    if saved > 0:
        print(f"\n✓ 成功提取 {saved} 帧到 {args.output}")
        return 0
    else:
        print("\n提示：请转换.bag文件为视频格式后再处理")
        return 1


if __name__ == "__main__":
    sys.exit(main())
