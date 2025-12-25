#!/usr/bin/env python3
"""
视频文件处理节点
处理本地视频文件并生成标注视频

功能:
1. 读取本地视频文件
2. 逐帧进行YOLO推理
3. 生成带标注的输出视频
4. 支持批量处理多个视频
5. 生成检测统计报告

使用方法:
    # 处理单个视频
    python video_processor_node.py --input video.mp4 --output output.mp4
    
    # 批量处理目录下所有视频
    python video_processor_node.py --input videos/ --output results/
    
    # 仅生成统计报告
    python video_processor_node.py --input video.mp4 --stats-only
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

# 尝试导入ROS
try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class VideoProcessor:
    """视频文件处理器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化视频处理器
        
        Args:
            model_path: YOLO模型路径
        """
        # 模型路径
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "best.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 推理参数
        self.conf_threshold = 0.5
        self.imgsz = 640
        self.device = 'cpu'
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': {},
            'processing_time': 0,
        }
    
    def process_video(self, input_path: str, output_path: str = None,
                     show_progress: bool = True, save_frames: bool = False) -> Dict:
        """
        处理单个视频文件
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径 (None则不保存)
            show_progress: 是否显示进度
            save_frames: 是否保存检测帧
            
        Returns:
            统计信息字典
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {input_path}")
        
        print(f"\n处理视频: {input_path.name}")
        print("=" * 50)
        
        # 打开视频
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {input_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {duration:.1f} 秒")
        
        # 创建输出视频
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # 保存检测帧的目录
        frames_dir = None
        if save_frames:
            frames_dir = input_path.parent / f"{input_path.stem}_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计初始化
        stats = {
            'input_file': str(input_path),
            'output_file': str(output_path) if output_path else None,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'class_counts': {},
            'frame_detections': [],  # 每帧的检测数量
            'processing_time': 0,
            'avg_inference_time': 0,
        }
        
        # 处理每一帧
        frame_count = 0
        inference_times = []
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 推理
            t0 = time.perf_counter()
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) * 1000)
            
            result = results[0]
            annotated_frame = result.plot()
            
            # 统计检测结果
            num_detections = len(result.boxes)
            stats['total_detections'] += num_detections
            stats['frame_detections'].append(num_detections)
            
            # 统计各类别数量
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                if cls_name not in stats['class_counts']:
                    stats['class_counts'][cls_name] = 0
                stats['class_counts'][cls_name] += 1
            
            # 写入输出视频
            if out is not None:
                out.write(annotated_frame)
            
            # 保存检测帧
            if save_frames and num_detections > 0:
                frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated_frame)
            
            frame_count += 1
            
            # 显示进度
            if show_progress and frame_count % 30 == 0:
                progress = frame_count / total_frames * 100 if total_frames > 0 else 0
                avg_time = np.mean(inference_times[-30:])
                print(f"  进度: {progress:.1f}% ({frame_count}/{total_frames}) | "
                      f"推理: {avg_time:.1f}ms | "
                      f"检测: {stats['total_detections']}")
        
        # 清理
        cap.release()
        if out is not None:
            out.release()
        
        # 最终统计
        end_time = time.time()
        stats['total_frames'] = total_frames
        stats['processed_frames'] = frame_count
        stats['processing_time'] = end_time - start_time
        stats['avg_inference_time'] = np.mean(inference_times) if inference_times else 0
        
        # 打印统计
        print("\n处理完成!")
        print(f"  处理帧数: {frame_count}")
        print(f"  检测总数: {stats['total_detections']}")
        print(f"  各类别统计: {stats['class_counts']}")
        print(f"  平均推理时间: {stats['avg_inference_time']:.1f} ms")
        print(f"  总处理时间: {stats['processing_time']:.1f} 秒")
        print(f"  处理速度: {frame_count / stats['processing_time']:.1f} FPS")
        
        if output_path:
            print(f"  输出文件: {output_path}")
        
        return stats
    
    def process_directory(self, input_dir: str, output_dir: str,
                         extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')) -> List[Dict]:
        """
        批量处理目录下的所有视频
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            extensions: 支持的视频格式
            
        Returns:
            所有视频的统计信息列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有视频文件
        video_files = []
        for ext in extensions:
            video_files.extend(input_dir.glob(f"*{ext}"))
            video_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        video_files = sorted(set(video_files))
        
        if not video_files:
            print(f"警告: 在 {input_dir} 中未找到视频文件")
            return []
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        all_stats = []
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] ", end="")
            
            output_path = output_dir / f"{video_path.stem}_annotated.mp4"
            
            try:
                stats = self.process_video(str(video_path), str(output_path))
                all_stats.append(stats)
            except Exception as e:
                print(f"处理失败: {e}")
                all_stats.append({'input_file': str(video_path), 'error': str(e)})
        
        return all_stats
    
    def generate_report(self, stats_list: List[Dict], output_path: str = None) -> str:
        """
        生成检测统计报告
        
        Args:
            stats_list: 统计信息列表
            output_path: 报告输出路径
            
        Returns:
            报告内容
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 汇总统计
        total_frames = sum(s.get('processed_frames', 0) for s in stats_list)
        total_detections = sum(s.get('total_detections', 0) for s in stats_list)
        total_time = sum(s.get('processing_time', 0) for s in stats_list)
        
        # 合并类别统计
        all_class_counts = {}
        for s in stats_list:
            for cls, count in s.get('class_counts', {}).items():
                if cls not in all_class_counts:
                    all_class_counts[cls] = 0
                all_class_counts[cls] += count
        
        report = f"""# 障碍物检测统计报告

生成时间: {timestamp}

## 概览

| 指标 | 数值 |
|------|------|
| 处理视频数 | {len(stats_list)} |
| 总帧数 | {total_frames:,} |
| 检测总数 | {total_detections:,} |
| 总处理时间 | {total_time:.1f} 秒 |
| 平均处理速度 | {total_frames / total_time:.1f} FPS |

## 类别统计

| 类别 | 检测数量 | 占比 |
|------|----------|------|
"""
        for cls, count in sorted(all_class_counts.items(), key=lambda x: -x[1]):
            ratio = count / total_detections * 100 if total_detections > 0 else 0
            report += f"| {cls} | {count:,} | {ratio:.1f}% |\n"
        
        report += f"""
## 详细统计

"""
        for i, s in enumerate(stats_list, 1):
            if 'error' in s:
                report += f"### {i}. {Path(s['input_file']).name} (错误)\n"
                report += f"错误信息: {s['error']}\n\n"
            else:
                report += f"### {i}. {Path(s['input_file']).name}\n"
                report += f"- 分辨率: {s.get('resolution', 'N/A')}\n"
                report += f"- 处理帧数: {s.get('processed_frames', 0):,}\n"
                report += f"- 检测数量: {s.get('total_detections', 0):,}\n"
                report += f"- 平均推理: {s.get('avg_inference_time', 0):.1f} ms\n"
                report += f"- 类别分布: {s.get('class_counts', {})}\n\n"
        
        report += f"""
## 模型信息

- 模型: {self.model.model_name}
- 置信度阈值: {self.conf_threshold}
- 输入尺寸: {self.imgsz}
- 推理设备: {self.device}

---
*由扫地机器人障碍物检测系统生成*
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存: {output_path}")
        
        return report


class VideoProcessorNode:
    """ROS视频处理节点（兼容ROS环境）"""
    
    def __init__(self, model_path: str = None):
        """初始化ROS节点"""
        if ROS_AVAILABLE:
            rospy.init_node('video_processor_demo', anonymous=True)
            rospy.loginfo("视频处理节点已启动")
        
        self.processor = VideoProcessor(model_path)
    
    def process_and_publish(self, input_path: str, output_path: str):
        """处理视频并发布进度（ROS模式）"""
        stats = self.processor.process_video(input_path, output_path)
        
        if ROS_AVAILABLE:
            rospy.loginfo(f"处理完成: {stats['total_detections']} 个检测")
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="视频文件处理节点 - 处理本地视频并生成标注结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 处理单个视频
    python video_processor_node.py --input demo.mp4 --output demo_annotated.mp4
    
    # 批量处理目录
    python video_processor_node.py --input videos/ --output results/
    
    # 仅生成报告
    python video_processor_node.py --input demo.mp4 --stats-only --report stats.md
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入视频文件或目录')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出视频文件或目录')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='YOLO模型路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='cpu',
                       help='推理设备 (cpu/cuda/mps)')
    parser.add_argument('--stats-only', action='store_true',
                       help='仅计算统计，不保存视频')
    parser.add_argument('--save-frames', action='store_true',
                       help='保存检测帧')
    parser.add_argument('--report', '-r', type=str, default=None,
                       help='生成统计报告的路径')
    
    args = parser.parse_args()
    
    try:
        processor = VideoProcessor(args.model)
        processor.conf_threshold = args.conf
        processor.device = args.device
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 处理单个视频
            output_path = args.output if not args.stats_only else None
            stats = processor.process_video(
                str(input_path),
                output_path,
                save_frames=args.save_frames
            )
            stats_list = [stats]
            
        elif input_path.is_dir():
            # 批量处理目录
            output_dir = args.output or str(input_path / "annotated")
            if args.stats_only:
                output_dir = None
            stats_list = processor.process_directory(str(input_path), output_dir)
            
        else:
            print(f"错误: 输入路径不存在: {input_path}")
            return 1
        
        # 生成报告
        if args.report:
            processor.generate_report(stats_list, args.report)
        elif args.stats_only:
            report = processor.generate_report(stats_list)
            print("\n" + report)
        
        # 保存JSON统计
        if args.report:
            json_path = Path(args.report).with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                # 移除不可序列化的字段
                for s in stats_list:
                    s.pop('frame_detections', None)
                json.dump(stats_list, f, indent=2, ensure_ascii=False)
            print(f"JSON统计已保存: {json_path}")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

