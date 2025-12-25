#!/usr/bin/env python3
"""
从开源数据库（COCO、Open Images 等）下载和筛选相关的障碍物图像
重点关注：地板、线缆、袜子、鞋子、玩具等
"""
import os
import sys
import json
from pathlib import Path
import random

print("""
=== 开源数据集集成建议 ===

由于许可证和网络限制，建议采用以下策略：

1. **COCO 数据集** (最推荐)
   - 下载: https://cocodataset.org/
   - 相关类别: sock, shoe, toy, tie, backpack, handbag
   - 命令: 
     ```
     cd data/
     wget http://images.cocodataset.org/zips/train2017.zip  # 19GB
     unzip train2017.zip
     ```

2. **Open Images** (替代方案)
   - 下载: https://storage.googleapis.com/openimages/web/index.html
   - 优点: 更多生活场景，包含电线/电缆
   - 工具: downloader https://github.com/EscVM/OIDv4_ToolKit

3. **本地数据扩增** (立即可用)
   - 使用你现有的 497 张 ascamera_nuwa_1 图像
   - 通过 augment_for_vacuum.py 生成 1500+ 张变体

推荐步骤：
========

【第一阶段】立即执行 - 使用本地数据扩增
  python scripts/augment_for_vacuum.py \\
    --input data/seed_dataset_v2 \\
    --output data/augmented_dataset \\
    --multiplier 5 \\
    --intensity 0.5

  结果: 200 张 × 5 = 1000 张高质量训练数据

【第二阶段】集成开源数据（可选）
  1. 下载 COCO 的子集（约 5000 张相关图像）
  2. 运行此脚本的 filtering 功能
  3. 应用相同的增强管道

【第三阶段】混合训练
  - 70% 本地数据（原始 200 + 扩增 800）
  - 30% 开源数据（筛选的相关类别）

当前脚本功能：
"""
)

def filter_coco_dataset(coco_json, output_dir, target_count=5000):
    """
    从 COCO JSON 中筛选相关的障碍物图像
    相关类别: sock, shoe, teddy bear, tie, cable/wire
    """
    print("\n⚠ 此功能需要 COCO 数据集已下载")
    print("  步骤: 1. 下载 COCO 数据集")
    print("       2. 解析 instances_train2017.json")
    print("       3. 筛选相关类别的图像")
    
    if not Path(coco_json).exists():
        print(f"\n❌ 未找到 {coco_json}")
        print("   请先下载 COCO 数据集")
        return 0
    
    with open(coco_json) as f:
        coco = json.load(f)
    
    # 相关的 COCO 类别 ID
    relevant_categories = {
        24: 'backpack',   # 背包 - 可能包含电线
        27: 'tie',        # 领带 - 类似细长物体
        32: 'shoe',       # 鞋子
        33: 'sock',       # 袜子
        39: 'teddy bear', # 玩具熊
    }
    
    relevant_cat_ids = set(relevant_categories.keys())
    
    # 筛选图像
    selected_images = []
    for ann in coco['annotations']:
        if ann['category_id'] in relevant_cat_ids:
            img_id = ann['image_id']
            if not any(img['id'] == img_id for img in selected_images):
                img_info = next((img for img in coco['images'] if img['id'] == img_id), None)
                if img_info:
                    selected_images.append(img_info)
    
    # 随机选择指定数量
    selected = random.sample(selected_images, min(target_count, len(selected_images)))
    
    print(f"\n✓ 筛选出 {len(selected)} 张相关图像")
    for cat_id, cat_name in relevant_categories.items():
        count = sum(1 for ann in coco['annotations'] 
                   if ann['category_id'] == cat_id and 
                   any(img['id'] == ann['image_id'] for img in selected))
        print(f"  {cat_name:15s}: {count:4d} 张")
    
    return len(selected)

def create_augmentation_strategy():
    """生成增强策略配置"""
    strategy = {
        "local_data": {
            "source": "data/seed_dataset_v2",
            "augmentation_multiplier": 5,
            "intensity": 0.5,
            "output": "data/augmented_dataset",
            "expected_count": 1000
        },
        "open_source_data": {
            "coco": {
                "enabled": False,  # 需要手动下载
                "source": "data/coco/train2017",
                "json": "data/coco/annotations/instances_train2017.json",
                "target_classes": ["sock", "shoe", "teddy bear", "tie"],
                "count": 3000
            },
            "open_images": {
                "enabled": False,  # 需要手动下载
                "source": "data/open_images",
                "classes": ["Wire", "Cable", "Sock", "Shoe", "Toy"],
                "count": 2000
            }
        },
        "training_split": {
            "local_augmented": 0.7,
            "open_source": 0.3,
            "total_expected": 2500
        }
    }
    
    config_path = Path('data/augmentation_strategy.json')
    with open(config_path, 'w') as f:
        json.dump(strategy, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 增强策略已保存到: {config_path}")
    return strategy

if __name__ == '__main__':
    print("\n" + "=" * 70)
    create_augmentation_strategy()
    print("=" * 70)
    
    print("\n【立即可执行的命令】\n")
    print("1. 生成 1000 张本地扩增数据:")
    print("   python scripts/augment_for_vacuum.py --multiplier 5 --intensity 0.5\n")
    
    print("2. 检查结果:")
    print("   ls -lh data/augmented_dataset/ | wc -l\n")
    
    print("3. 为训练做准备:")
    print("   python scripts/prepare_training_data.py \\")
    print("     --source data/augmented_dataset \\")
    print("     --output data/training_data \\")
    print("     --split 0.8\n")
