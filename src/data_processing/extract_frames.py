"""
RGBD视频帧提取工具
从RGBD视频中提取RGB和Depth帧，用于后续标注和训练
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

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("警告: pyrealsense2未安装，无法处理.bag格式文件")


def extract_frames_from_video(video_path, output_dir, sample_rate=2):
    """
    从普通视频文件提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        sample_rate: 采样率（每N帧提取一帧）
    """
    print(f"正在处理视频: {video_path}")
    
    # 创建输出目录
    rgb_dir = Path(output_dir) / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return 0
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {fps:.2f} FPS, 总帧数: {total_frames}")
    
    # 提取帧
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按采样率提取
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


def extract_frames_from_realsense_bag(bag_path, output_dir, sample_rate=2):
    """
    从RealSense .bag文件提取RGB和Depth帧
    
    Args:
        bag_path: .bag文件路径
        output_dir: 输出目录
        sample_rate: 采样率
    """
    if not REALSENSE_AVAILABLE:
        print("错误: pyrealsense2未安装，无法处理.bag文件")
        return 0
    
    print(f"正在处理RealSense录制: {bag_path}")
    
    # 创建输出目录
    rgb_dir = Path(output_dir) / "rgb"
    depth_dir = Path(output_dir) / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 配置RealSense管道
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, str(bag_path))
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        
        # 启动管道
        pipeline.start(config)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                
                # 按采样率提取
                if frame_count % sample_rate == 0:
                    # RGB帧
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                        rgb_path = rgb_dir / f"frame_{saved_count:06d}.jpg"
                        cv2.imwrite(str(rgb_path), color_image)
                    
                    # Depth帧
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_path = depth_dir / f"frame_{saved_count:06d}.png"
                        cv2.imwrite(str(depth_path), depth_image)
                    
                    saved_count += 1
                    
                    if saved_count % 100 == 0:
                        print(f"已提取 {saved_count} 帧...")
                
                frame_count += 1
                
            except RuntimeError:
                # 到达文件末尾
                break
        
        pipeline.stop()
        print(f"✓ 完成！共提取 {saved_count} 帧")
        print(f"  - RGB: {rgb_dir}")
        print(f"  - Depth: {depth_dir}")
        return saved_count
        
    except Exception as e:
        print(f"错误: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="从RGBD视频提取帧")
    parser.add_argument("input", help="输入视频文件路径（支持.mp4, .avi, .bag等）")
    parser.add_argument("--output", "-o", default="data/frames", 
                       help="输出目录 (默认: data/frames)")
    parser.add_argument("--sample-rate", "-s", type=int, default=2,
                       help="采样率，每N帧提取一帧 (默认: 2)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在 {args.input}")
        return 1
    
    # 根据文件类型选择处理方法
    file_ext = input_path.suffix.lower()
    
    if file_ext == ".bag":
        saved = extract_frames_from_realsense_bag(
            args.input, args.output, args.sample_rate
        )
    else:
        saved = extract_frames_from_video(
            args.input, args.output, args.sample_rate
        )
    
    if saved > 0:
        print(f"\n成功提取 {saved} 帧到 {args.output}")
        return 0
    else:
        print("\n提取失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
