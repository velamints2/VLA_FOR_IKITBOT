"""
数据集划分工具
将标注好的数据集划分为训练集和验证集
"""
import os
import random
import argparse
from pathlib import Path


def split_train_val(image_dir, label_dir, output_dir, train_ratio=0.8, seed=42):
    """
    划分训练集和验证集
    
    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        output_dir: 输出目录（生成train.txt和val.txt）
        train_ratio: 训练集比例
        seed: 随机种子
    """
    random.seed(seed)
    
    # 获取所有图像文件
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 收集所有有对应标签的图像
    valid_images = []
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            # 检查是否有对应的标签文件
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_images.append(img_path)
    
    if not valid_images:
        print("错误: 未找到有效的图像-标签对")
        return False
    
    print(f"找到 {len(valid_images)} 个有效样本")
    
    # 随机打乱
    random.shuffle(valid_images)
    
    # 划分
    split_idx = int(len(valid_images) * train_ratio)
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]
    
    print(f"训练集: {len(train_images)} 样本")
    print(f"验证集: {len(val_images)} 样本")
    
    # 写入train.txt
    train_file = output_dir / "train.txt"
    with open(train_file, 'w') as f:
        for img_path in train_images:
            f.write(f"{img_path.absolute()}\n")
    
    # 写入val.txt
    val_file = output_dir / "val.txt"
    with open(val_file, 'w') as f:
        for img_path in val_images:
            f.write(f"{img_path.absolute()}\n")
    
    print(f"✓ 划分完成！")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="划分训练集和验证集")
    parser.add_argument("--images", required=True, help="图像目录路径")
    parser.add_argument("--labels", required=True, help="标签目录路径")
    parser.add_argument("--output", "-o", default="data/seed_dataset",
                       help="输出目录 (默认: data/seed_dataset)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="训练集比例 (默认: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    success = split_train_val(
        args.images,
        args.labels,
        args.output,
        args.train_ratio,
        args.seed
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
