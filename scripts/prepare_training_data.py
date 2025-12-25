#!/usr/bin/env python3
"""
为训练准备数据：组织成 train/val/test 拆分，支持数据混合
"""
import shutil
from pathlib import Path
import argparse
from collections import defaultdict
import random

def prepare_training_data(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    组织数据集为 train/val/test 拆分
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像（分组 - 每个源图像的所有变体）
    images = sorted(source_path.glob('*.jpg')) + sorted(source_path.glob('*.png'))
    
    # 按原始图像分组
    image_groups = defaultdict(list)
    for img_path in images:
        # 提取原始名称（去掉 _aug 和 _orig）
        base_name = img_path.stem
        if '_aug' in base_name or '_orig' in base_name:
            original = base_name.split('_aug')[0].split('_orig')[0]
        else:
            original = base_name
        image_groups[original].append(img_path)
    
    print(f"✓ 找到 {len(image_groups)} 个图像组 ({len(images)} 个文件)")
    
    # 按组进行拆分（确保同一源图像的变体在同一集合中）
    unique_originals = list(image_groups.keys())
    random.shuffle(unique_originals)
    
    train_count = int(len(unique_originals) * train_ratio)
    val_count = int(len(unique_originals) * val_ratio)
    
    train_originals = unique_originals[:train_count]
    val_originals = unique_originals[train_count:train_count + val_count]
    test_originals = unique_originals[train_count + val_count:]
    
    # 复制文件
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for original in train_originals:
        for img_path in image_groups[original]:
            dest = output_path / 'train' / 'images' / img_path.name
            shutil.copy2(img_path, dest)
            stats['train'] += 1
    
    for original in val_originals:
        for img_path in image_groups[original]:
            dest = output_path / 'val' / 'images' / img_path.name
            shutil.copy2(img_path, dest)
            stats['val'] += 1
    
    for original in test_originals:
        for img_path in image_groups[original]:
            dest = output_path / 'test' / 'images' / img_path.name
            shutil.copy2(img_path, dest)
            stats['test'] += 1
    
    print(f"\n数据拆分完成:")
    print(f"  Train: {stats['train']} 张 ({100*stats['train']/len(images):.1f}%)")
    print(f"  Val:   {stats['val']:3d} 张 ({100*stats['val']/len(images):.1f}%)")
    print(f"  Test:  {stats['test']:3d} 张 ({100*stats['test']/len(images):.1f}%)")
    
    # 生成 YAML 配置
    yaml_content = f"""path: {output_path.absolute()}
train: train/images
val: val/images
test: test/images

nc: 6
names: ['wire', 'slipper', 'sock', 'cable', 'toy', 'obstacle']
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ 数据集配置已保存到: {yaml_path}")
    print(f"\n用于 YOLO 训练:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=100")
    
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为训练准备数据集')
    parser.add_argument('--source', default='data/augmented_dataset', help='源数据目录')
    parser.add_argument('--output', default='data/training_data', help='输出目录')
    parser.add_argument('--train', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val', type=float, default=0.1, help='验证集比例')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("准备训练数据集")
    print("=" * 70 + "\n")
    
    prepare_training_data(args.source, args.output, args.train, args.val)
    
    print("\n" + "=" * 70)
    print("✓ 准备完成，可以开始训练")
    print("=" * 70)
