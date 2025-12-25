#!/usr/bin/env python3
"""
检查 Label Studio 标注进度
"""
import os
import json
from pathlib import Path

def check_annotation_progress():
    """检查标注进度"""
    
    # 检查数据目录
    dataset_dir = Path("data/seed_dataset_v2")
    auto_labels_dir = dataset_dir / "auto_labels"
    
    print("=" * 60)
    print("📊 标注进度检查")
    print("=" * 60)
    
    # 统计图像文件
    image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
    print(f"\n📸 图像文件:")
    print(f"  - 总数: {len(image_files)}")
    print(f"  - 位置: {dataset_dir}")
    
    # 统计预标注文件
    auto_label_files = list(auto_labels_dir.glob("*.txt")) if auto_labels_dir.exists() else []
    print(f"\n🤖 预标注文件 (AI生成):")
    print(f"  - 总数: {len(auto_label_files)}")
    print(f"  - 位置: {auto_labels_dir}")
    print(f"  - 覆盖率: {len(auto_label_files)}/{len(image_files)} ({len(auto_label_files)/len(image_files)*100:.1f}%)")
    
    # 检查 Label Studio 导出目录
    ls_export_dir = Path("label_studio/data/export")
    if ls_export_dir.exists():
        export_files = list(ls_export_dir.glob("*.json"))
        print(f"\n📦 Label Studio 导出:")
        print(f"  - 导出文件数: {len(export_files)}")
        if export_files:
            latest_export = max(export_files, key=lambda p: p.stat().st_mtime)
            print(f"  - 最新导出: {latest_export.name}")
            print(f"  - 修改时间: {latest_export.stat().st_mtime}")
    
    # 检查标注文件（人工审核后）
    labels_dir = dataset_dir / "labels"
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        print(f"\n✅ 人工审核标注:")
        print(f"  - 总数: {len(label_files)}")
        print(f"  - 位置: {labels_dir}")
        print(f"  - 完成率: {len(label_files)}/{len(image_files)} ({len(label_files)/len(image_files)*100:.1f}%)")
    else:
        print(f"\n⚠️  人工审核标注目录不存在: {labels_dir}")
        print(f"  - 建议: 在 Label Studio 中完成标注后导出")
    
    print("\n" + "=" * 60)
    print("📋 下一步建议:")
    print("=" * 60)
    
    if len(auto_label_files) < len(image_files):
        missing_count = len(image_files) - len(auto_label_files)
        print(f"1. ⚠️  有 {missing_count} 张图像缺少预标注")
        print(f"   运行: python scripts/auto_annotate.py data/seed_dataset_v2")
    
    if not labels_dir.exists() or len(list(labels_dir.glob("*.txt"))) == 0:
        print(f"2. 🏷️  在 Label Studio 中审核标注:")
        print(f"   - 访问: http://localhost:8080")
        print(f"   - 审核并修正预标注")
        print(f"   - 完成后导出 YOLO 格式")
    
    print(f"\n3. 📚 查看文档:")
    print(f"   - 标注指南: docs/annotation_tools_guide.md")
    print(f"   - 快速参考: docs/ANNOTATION_QUICKREF.md")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_annotation_progress()
