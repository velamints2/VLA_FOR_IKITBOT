#!/usr/bin/env python3
"""
从 ascamera_nuwa_1 文件夹中重新选择 200 张图像作为种子训练集
"""
import shutil
from pathlib import Path
import random

# 源和目标目录
SOURCE_PATTERN = Path('/Users/macbookair/Documents/trae_projects/llm/data/frames').glob(
    '*/ascamera_nuwa_1_rgb0_image/*.jpg'
)
DEST_DIR = Path('/Users/macbookair/Documents/trae_projects/llm/data/seed_dataset_v2')
TARGET_COUNT = 200

# 获取所有 ascamera_nuwa_1 中的图像
source_images = sorted(list(SOURCE_PATTERN))
print(f"found {len(source_images)} images from ascamera_nuwa_1")

if len(source_images) < TARGET_COUNT:
    print(f"❌ 错误：只找到 {len(source_images)} 张图像，少于目标的 {TARGET_COUNT} 张")
    exit(1)

# 随机选择 200 张
random.seed(42)  # 保证可重复性
selected = random.sample(source_images, TARGET_COUNT)
selected = sorted(selected)

print(f"✓ 已随机选择 {len(selected)} 张图像")

# 清空目标目录（保留其他文件夹）
print("\n清理旧文件...", end='', flush=True)
for img in DEST_DIR.glob('*.jpg'):
    img.unlink()
for img in DEST_DIR.glob('*.png'):
    img.unlink()
print(" ✓")

# 复制新的图像，重新命名为 seed_XXXX.jpg
print("\n复制图像...", end='', flush=True)
for idx, src in enumerate(selected):
    dest = DEST_DIR / f'seed_{idx:04d}.jpg'
    shutil.copy2(src, dest)
    if (idx + 1) % 50 == 0:
        print(f"\n  [{idx + 1:3d}/{TARGET_COUNT}]", end='', flush=True)

print(f"\n✓ 已完成\n")
print(f"=== 结果 ===")
print(f"✓ 已从 ascamera_nuwa_1 选择 {TARGET_COUNT} 张图像")
print(f"✓ 保存到：{DEST_DIR}")
print(f"\n来源分布：")

# 统计来源 bag 文件
from collections import Counter
source_bags = Counter([str(p.parent.parent.name) for p in selected])
for bag, count in sorted(source_bags.items()):
    print(f"  {bag}: {count} 张")
