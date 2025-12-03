#!/usr/bin/env python3
"""
YOLO数据集准备脚本
将种子数据集划分为训练集和验证集，组织成YOLO格式

YOLO数据集结构:
data/seed_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       └── ...
└── classes.txt
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from glob import glob


def prepare_yolo_dataset(
    source_dir: str,
    output_dir: str = None,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    准备YOLO格式数据集
    
    Args:
        source_dir: 源数据目录（包含 images/ 和 labels/）
        output_dir: 输出目录（默认为source_dir）
        train_ratio: 训练集比例
        seed: 随机种子
    """
    random.seed(seed)
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir) if output_dir else source_dir
    
    # 源目录
    src_images_dir = source_dir / "images"
    src_labels_dir = source_dir / "labels"
    
    # 检查源目录
    if not src_images_dir.exists():
        print(f"错误: 图像目录不存在 {src_images_dir}")
        return False
    
    # 收集所有图像
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(src_images_dir.glob(f"*{ext}"))
        all_images.extend(src_images_dir.glob(f"*{ext.upper()}"))
    
    # 去重并排序
    all_images = sorted(set(all_images))
    
    if not all_images:
        print(f"错误: 在 {src_images_dir} 中未找到图像")
        return False
    
    print(f"找到 {len(all_images)} 张图像")
    
    # 检查标签文件
    labeled_images = []
    unlabeled_images = []
    
    for img_path in all_images:
        label_path = src_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            labeled_images.append(img_path)
        else:
            unlabeled_images.append(img_path)
    
    print(f"已标注: {len(labeled_images)} 张")
    print(f"未标注: {len(unlabeled_images)} 张")
    
    if not labeled_images:
        print("\n警告: 没有找到标注文件！")
        print("请先使用 LabelImg 进行标注：")
        print(f"  labelImg {src_images_dir} {source_dir}/classes.txt {src_labels_dir}")
        
        # 创建示例标注（用于流程验证）
        print("\n是否创建示例标注用于流程验证？")
        response = input("输入 'yes' 创建示例标注: ").strip().lower()
        if response == 'yes':
            create_sample_labels(all_images[:10], src_labels_dir)
            labeled_images = all_images[:10]
        else:
            return False
    
    # 打乱并划分
    random.shuffle(labeled_images)
    split_idx = int(len(labeled_images) * train_ratio)
    train_images = labeled_images[:split_idx]
    val_images = labeled_images[split_idx:]
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_images)} 张 ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_images)} 张 ({(1-train_ratio)*100:.0f}%)")
    
    # 创建YOLO目录结构
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 复制/移动文件
    print("\n组织文件...")
    
    def copy_files(images, img_dst, label_dst):
        for img_path in images:
            # 复制图像
            shutil.copy2(img_path, img_dst / img_path.name)
            
            # 复制标签
            label_path = src_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, label_dst / label_path.name)
    
    copy_files(train_images, train_img_dir, train_label_dir)
    copy_files(val_images, val_img_dir, val_label_dir)
    
    # 验证
    train_img_count = len(list(train_img_dir.glob("*")))
    val_img_count = len(list(val_img_dir.glob("*")))
    train_label_count = len(list(train_label_dir.glob("*.txt")))
    val_label_count = len(list(val_label_dir.glob("*.txt")))
    
    print(f"\n✓ 数据集准备完成！")
    print(f"  训练集: {train_img_count} 图像, {train_label_count} 标签")
    print(f"  验证集: {val_img_count} 图像, {val_label_count} 标签")
    print(f"\n目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({train_img_count})")
    print(f"  │   └── val/ ({val_img_count})")
    print(f"  └── labels/")
    print(f"      ├── train/ ({train_label_count})")
    print(f"      └── val/ ({val_label_count})")
    
    return True


def create_sample_labels(images, labels_dir, num_classes=6):
    """
    创建示例标注文件（用于流程验证）
    
    YOLO标注格式: class_id center_x center_y width height (归一化到0-1)
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n创建 {len(images)} 个示例标注...")
    
    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # 生成1-3个随机框
        num_boxes = random.randint(1, 3)
        lines = []
        
        for _ in range(num_boxes):
            class_id = random.randint(0, num_classes - 1)
            # 随机中心点
            cx = random.uniform(0.2, 0.8)
            cy = random.uniform(0.2, 0.8)
            # 随机宽高
            w = random.uniform(0.05, 0.3)
            h = random.uniform(0.05, 0.3)
            
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))
    
    print(f"✓ 示例标注已创建到 {labels_dir}")


def main():
    parser = argparse.ArgumentParser(description="准备YOLO格式数据集")
    parser.add_argument('source_dir', help='源数据目录')
    parser.add_argument('--output', '-o', help='输出目录（默认为源目录）')
    parser.add_argument('--train-ratio', '-r', type=float, default=0.8,
                       help='训练集比例（默认0.8）')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--create-samples', action='store_true',
                       help='创建示例标注用于流程验证')
    
    args = parser.parse_args()
    
    if args.create_samples:
        # 仅创建示例标注
        src_images_dir = Path(args.source_dir) / "images"
        src_labels_dir = Path(args.source_dir) / "labels"
        
        images = list(src_images_dir.glob("*.jpg")) + list(src_images_dir.glob("*.png"))
        create_sample_labels(images[:20], src_labels_dir)
    else:
        prepare_yolo_dataset(
            args.source_dir,
            args.output,
            args.train_ratio,
            args.seed
        )


if __name__ == "__main__":
    main()
