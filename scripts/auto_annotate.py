#!/usr/bin/env python3
"""
半自动标注脚本
使用预训练 YOLO 模型对图像进行预标注
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

try:
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    print(f"错误: 缺少依赖库 - {e}")
    print("请运行: pip install ultralytics opencv-python")
    sys.exit(1)


def auto_annotate(
    input_dir: str,
    output_dir: str,
    model_path: str = "yolo11n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save_txt: bool = True,
    save_conf: bool = True,
    visualize: bool = False
):
    """
    使用 YOLO 模型进行自动标注
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出标注目录
        model_path: YOLO 模型路径
        conf_threshold: 置信度阈值
        iou_threshold: IOU 阈值
        save_txt: 是否保存 YOLO 格式标注
        save_conf: 是否在标注中包含置信度
        visualize: 是否保存可视化结果
    """
    
    print("=" * 60)
    print("半自动标注工具")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print("=" * 60)
    print()
    
    # 加载模型
    print("加载模型...")
    try:
        model = YOLO(model_path)
        print(f"✓ 模型加载成功: {model_path}")
        print(f"  模型类别: {model.names}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 获取所有图像
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"✗ 在 {input_dir} 中未找到图像")
        return
    
    print(f"找到 {len(images)} 张图像")
    print()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)
    
    # 处理每张图像
    total_detections = 0
    processed = 0
    
    print("开始预标注...")
    for img_path in tqdm(images, desc="预标注进度"):
        try:
            # 运行推理
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) > 0:
                total_detections += len(boxes)
                
                # 保存 YOLO 格式标注
                if save_txt:
                    label_file = output_path / f"{img_path.stem}.txt"
                    
                    with open(label_file, 'w') as f:
                        for box in boxes:
                            # YOLO 格式: class x_center y_center width height [confidence]
                            cls = int(box.cls[0])
                            x, y, w, h = box.xywhn[0].tolist()
                            
                            if save_conf:
                                conf = float(box.conf[0])
                                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")
                            else:
                                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                
                # 保存可视化结果
                if visualize:
                    vis_img = result.plot()
                    vis_file = vis_path / f"{img_path.stem}_annotated.jpg"
                    cv2.imwrite(str(vis_file), vis_img)
            
            processed += 1
            
        except Exception as e:
            print(f"✗ 处理失败: {img_path.name} - {e}")
            continue
    
    print()
    print("=" * 60)
    print("✓ 预标注完成！")
    print("=" * 60)
    print(f"处理图像: {processed}/{len(images)}")
    print(f"总检测数: {total_detections}")
    print(f"平均每张: {total_detections/processed:.1f}" if processed > 0 else "N/A")
    print(f"输出目录: {output_path}")
    print()
    
    # 统计每个类别的检测数
    if total_detections > 0:
        print("类别分布:")
        label_files = list(output_path.glob("*.txt"))
        class_counts = {}
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            class_name = model.names.get(cls, f"class_{cls}")
            print(f"  {class_name}: {count} 个检测")
    
    print()
    print("下一步:")
    print(f"  1. 使用 LabelImg 检查标注:")
    print(f"     labelImg {input_dir} {output_dir}")
    print(f"  2. 或导入到 Label Studio 进行团队审核")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="使用 YOLO 进行半自动标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python auto_annotate.py --input data/seed_dataset_v2 --output data/auto_labels
  
  # 指定模型和阈值
  python auto_annotate.py --input data/seed_dataset_v2 --model yolo11n.pt --conf 0.3
  
  # 启用可视化
  python auto_annotate.py --input data/seed_dataset_v2 --visualize
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入图像目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出标注目录'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='yolo11n.pt',
        help='YOLO 模型路径 (默认: yolo11n.pt)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='置信度阈值 (默认: 0.25)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU 阈值 (默认: 0.45)'
    )
    
    parser.add_argument(
        '--save-txt',
        action='store_true',
        default=True,
        help='保存 YOLO 格式标注文件'
    )
    
    parser.add_argument(
        '--save-conf',
        action='store_true',
        default=True,
        help='在标注中包含置信度'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='保存可视化结果'
    )
    
    args = parser.parse_args()
    
    auto_annotate(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
