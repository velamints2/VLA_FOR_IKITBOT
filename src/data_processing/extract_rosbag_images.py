#!/usr/bin/env python3
"""
ROS Bag 图像提取工具
从 ROS bag 文件中提取 sensor_msgs/Image 类型的图像数据

使用: python extract_rosbag_images.py <bag_file> <output_dir> [sample_rate]
"""

import os
import sys
import glob
import numpy as np
import cv2
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


def extract_images_from_rosbag(bag_file: str, output_dir: str, sample_rate: int = 1):
    """
    从 ROS bag 文件提取图像
    
    Args:
        bag_file: .bag 文件路径
        output_dir: 输出目录
        sample_rate: 采样率（每 N 帧取一帧）
    """
    print("=" * 60)
    print("ROS Bag 图像提取工具")
    print("=" * 60)
    print(f"输入文件: {bag_file}")
    print(f"输出目录: {output_dir}")
    print(f"采样率: 每 {sample_rate} 帧提取一帧")
    print("=" * 60)
    print()
    
    # 创建类型存储
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    with Reader(bag_file) as reader:
        # 找到所有图像 topic
        image_topics = []
        for conn in reader.connections:
            if 'Image' in conn.msgtype:
                image_topics.append((conn.topic, conn.msgtype, conn.msgcount))
                print(f"发现图像 topic: {conn.topic} ({conn.msgcount} 帧)")
        
        if not image_topics:
            print("错误: 未找到图像数据")
            return
        
        print()
        
        # 为每个 topic 创建输出目录
        topic_dirs = {}
        for topic, _, _ in image_topics:
            # 清理 topic 名称作为目录名
            clean_name = topic.replace('/', '_').strip('_')
            topic_dir = os.path.join(output_dir, clean_name)
            os.makedirs(topic_dir, exist_ok=True)
            topic_dirs[topic] = topic_dir
            print(f"输出目录: {topic_dir}")
        
        print()
        print("开始提取图像...")
        
        # 统计每个 topic 的帧计数
        topic_counts = {topic: 0 for topic, _, _ in image_topics}
        topic_saved = {topic: 0 for topic, _, _ in image_topics}
        
        # 遍历所有消息
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            
            if topic not in topic_dirs:
                continue
            
            # 采样
            topic_counts[topic] += 1
            if topic_counts[topic] % sample_rate != 0:
                continue
            
            try:
                # 反序列化消息
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                
                # 获取图像参数
                height = msg.height
                width = msg.width
                encoding = msg.encoding
                data = np.frombuffer(msg.data, dtype=np.uint8)
                
                # 根据编码格式转换
                if encoding in ['rgb8', 'RGB8']:
                    img = data.reshape((height, width, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding in ['bgr8', 'BGR8']:
                    img = data.reshape((height, width, 3))
                elif encoding in ['mono8', 'MONO8']:
                    img = data.reshape((height, width))
                elif encoding in ['16UC1']:
                    data = np.frombuffer(msg.data, dtype=np.uint16)
                    img = data.reshape((height, width))
                elif encoding in ['32FC1']:
                    data = np.frombuffer(msg.data, dtype=np.float32)
                    img = data.reshape((height, width))
                    # 归一化到 0-255
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    img = img.astype(np.uint8)
                else:
                    # 尝试通用处理
                    channels = len(data) // (height * width)
                    if channels == 3:
                        img = data.reshape((height, width, 3))
                    elif channels == 1:
                        img = data.reshape((height, width))
                    else:
                        print(f"警告: 未知编码 {encoding}, 跳过")
                        continue
                
                # 保存图像
                filename = f"frame_{topic_saved[topic]:06d}.jpg"
                filepath = os.path.join(topic_dirs[topic], filename)
                cv2.imwrite(filepath, img)
                topic_saved[topic] += 1
                
                # 进度输出
                total_saved = sum(topic_saved.values())
                if total_saved % 50 == 0:
                    print(f"已提取 {total_saved} 帧...")
                    
            except Exception as e:
                print(f"警告: 处理帧时出错 - {e}")
                continue
        
        print()
        print("=" * 60)
        print("✓ 提取完成！")
        print("=" * 60)
        for topic, count in topic_saved.items():
            clean_name = topic.replace('/', '_').strip('_')
            print(f"{clean_name}: {count} 帧")
        print(f"总计: {sum(topic_saved.values())} 帧")
        print("=" * 60)


def batch_extract(raw_dir: str, output_dir: str, sample_rate: int = 1):
    """批量处理多个 bag 文件"""
    bag_files = sorted(glob.glob(os.path.join(raw_dir, "*.bag")))
    
    if not bag_files:
        print(f"错误: {raw_dir} 目录下没有找到 .bag 文件")
        return
    
    print(f"发现 {len(bag_files)} 个 bag 文件")
    print()
    
    for bag_file in bag_files:
        bag_name = Path(bag_file).stem
        bag_output_dir = os.path.join(output_dir, bag_name)
        
        print(f"\n{'#' * 60}")
        print(f"处理: {bag_file}")
        print(f"{'#' * 60}\n")
        
        try:
            extract_images_from_rosbag(bag_file, bag_output_dir, sample_rate)
        except Exception as e:
            print(f"错误: 处理 {bag_file} 失败 - {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✓ 所有文件处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  单文件: python extract_rosbag_images.py <bag_file> <output_dir> [sample_rate]")
        print("  批量:   python extract_rosbag_images.py --batch <raw_dir> <output_dir> [sample_rate]")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("批量模式需要指定 raw_dir 和 output_dir")
            sys.exit(1)
        raw_dir = sys.argv[2]
        output_dir = sys.argv[3]
        sample_rate = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
        batch_extract(raw_dir, output_dir, sample_rate)
    else:
        bag_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "output_frames"
        sample_rate = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
        extract_images_from_rosbag(bag_file, output_dir, sample_rate)
