"""
数据分析工具
分析RGBD图像数据集的统计特性
"""
import os
import sys
import argparse
from pathlib import Path
import json

try:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("请运行: pip install opencv-python numpy matplotlib")
    sys.exit(1)


def analyze_image_statistics(image_dir, output_dir=None, max_samples=1000):
    """
    分析图像统计特性
    
    Args:
        image_dir: 图像目录
        output_dir: 输出目录（保存分析报告和可视化）
        max_samples: 最大采样数量
    """
    image_dir = Path(image_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("图像数据集统计分析")
    print("=" * 60)
    
    # 收集图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"错误: 在 {image_dir} 中未找到图像文件")
        return None
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 采样（如果数量太多）
    if len(image_files) > max_samples:
        print(f"采样 {max_samples} 张进行分析...")
        import random
        image_files = random.sample(image_files, max_samples)
    
    # 统计变量
    resolutions = []
    brightness_values = []
    contrast_values = []
    aspect_ratios = []
    
    print("正在分析...")
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        resolutions.append((w, h))
        aspect_ratios.append(w / h)
        
        # 转灰度图计算亮度和对比度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray))
        contrast_values.append(np.std(gray))
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(image_files)} 张...")
    
    # 统计结果
    stats = {
        'total_images': len(image_files),
        'resolutions': {
            'unique_count': len(set(resolutions)),
            'most_common': max(set(resolutions), key=resolutions.count),
            'all': list(set(resolutions))
        },
        'brightness': {
            'mean': float(np.mean(brightness_values)),
            'std': float(np.std(brightness_values)),
            'min': float(np.min(brightness_values)),
            'max': float(np.max(brightness_values))
        },
        'contrast': {
            'mean': float(np.mean(contrast_values)),
            'std': float(np.std(contrast_values)),
            'min': float(np.min(contrast_values)),
            'max': float(np.max(contrast_values))
        },
        'aspect_ratio': {
            'mean': float(np.mean(aspect_ratios)),
            'std': float(np.std(aspect_ratios))
        }
    }
    
    # 打印结果
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总图像数: {stats['total_images']}")
    print(f"\n分辨率:")
    print(f"  - 不同分辨率数: {stats['resolutions']['unique_count']}")
    print(f"  - 最常见: {stats['resolutions']['most_common']}")
    print(f"\n亮度:")
    print(f"  - 平均: {stats['brightness']['mean']:.2f}")
    print(f"  - 标准差: {stats['brightness']['std']:.2f}")
    print(f"  - 范围: [{stats['brightness']['min']:.2f}, {stats['brightness']['max']:.2f}]")
    print(f"\n对比度:")
    print(f"  - 平均: {stats['contrast']['mean']:.2f}")
    print(f"  - 标准差: {stats['contrast']['std']:.2f}")
    print(f"\n宽高比:")
    print(f"  - 平均: {stats['aspect_ratio']['mean']:.2f}")
    
    # 保存结果
    if output_dir:
        # JSON报告
        report_path = output_dir / "data_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ 报告已保存: {report_path}")
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 亮度分布
        axes[0, 0].hist(brightness_values, bins=50, color='blue', alpha=0.7)
        axes[0, 0].set_title('Brightness Distribution')
        axes[0, 0].set_xlabel('Brightness')
        axes[0, 0].set_ylabel('Frequency')
        
        # 对比度分布
        axes[0, 1].hist(contrast_values, bins=50, color='green', alpha=0.7)
        axes[0, 1].set_title('Contrast Distribution')
        axes[0, 1].set_xlabel('Contrast (Std Dev)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 宽高比分布
        axes[1, 0].hist(aspect_ratios, bins=50, color='red', alpha=0.7)
        axes[1, 0].set_title('Aspect Ratio Distribution')
        axes[1, 0].set_xlabel('Width / Height')
        axes[1, 0].set_ylabel('Frequency')
        
        # 分辨率分布
        res_counts = {}
        for res in resolutions:
            res_str = f"{res[0]}x{res[1]}"
            res_counts[res_str] = res_counts.get(res_str, 0) + 1
        
        top_resolutions = sorted(res_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_resolutions:
            labels, values = zip(*top_resolutions)
            axes[1, 1].bar(range(len(labels)), values, color='purple', alpha=0.7)
            axes[1, 1].set_xticks(range(len(labels)))
            axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
            axes[1, 1].set_title('Top 10 Resolutions')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        viz_path = output_dir / "data_analysis_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化已保存: {viz_path}")
        plt.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="分析图像数据集统计特性")
    parser.add_argument("image_dir", help="图像目录路径")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="最大采样数 (默认: 1000)")
    
    args = parser.parse_args()
    
    stats = analyze_image_statistics(
        args.image_dir,
        args.output,
        args.max_samples
    )
    
    return 0 if stats else 1


if __name__ == "__main__":
    sys.exit(main())
