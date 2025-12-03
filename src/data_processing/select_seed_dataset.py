"""
快速图像预览和精选工具
用于从提取的帧中快速选择最具代表性的图像作为种子数据集
"""
import os
import sys
import argparse
from pathlib import Path
import shutil

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"错误: 缺少库 - {e}")
    print("请运行: conda activate obstacle_detection")
    sys.exit(1)


def select_diverse_frames(image_dir, output_dir, num_samples=100, method='uniform'):
    """
    从大量帧中选择多样化的样本
    
    Args:
        image_dir: 输入图像目录
        output_dir: 输出目录
        num_samples: 要选择的样本数
        method: 选择方法 ('uniform', 'random', 'histogram')
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有图像（递归搜索）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([f for f in image_dir.rglob('*') 
                         if f.suffix.lower() in image_extensions])
    
    if not image_files:
        print(f"错误: 在 {image_dir} 中未找到图像")
        return []
    
    total_images = len(image_files)
    print(f"找到 {total_images} 张图像")
    
    if total_images <= num_samples:
        print(f"图像数量少于 {num_samples}，复制全部图像")
        selected = image_files
    else:
        print(f"使用 {method} 方法选择 {num_samples} 张...")
        
        if method == 'uniform':
            # 均匀采样
            step = total_images / num_samples
            indices = [int(i * step) for i in range(num_samples)]
            selected = [image_files[i] for i in indices]
            
        elif method == 'random':
            # 随机采样
            import random
            random.seed(42)
            selected = random.sample(image_files, num_samples)
            
        elif method == 'histogram':
            # 基于直方图差异选择
            print("  计算图像直方图...")
            histograms = []
            for img_path in image_files:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    hist = cv2.calcHist([img], [0], None, [32], [0, 256])
                    hist = hist.flatten() / hist.sum()
                    histograms.append((img_path, hist))
            
            # 贪心选择：每次选择与已选图像差异最大的
            selected = [histograms[0][0]]  # 先选第一张
            histograms_selected = [histograms[0][1]]
            
            for _ in range(num_samples - 1):
                max_diff = -1
                max_idx = -1
                
                for i, (path, hist) in enumerate(histograms):
                    if path in selected:
                        continue
                    
                    # 计算与已选图像的最小差异
                    min_sim = min([np.sum(np.abs(hist - h)) 
                                  for h in histograms_selected])
                    
                    if min_sim > max_diff:
                        max_diff = min_sim
                        max_idx = i
                
                if max_idx >= 0:
                    selected.append(histograms[max_idx][0])
                    histograms_selected.append(histograms[max_idx][1])
                
                if (len(selected) % 10 == 0):
                    print(f"  已选择 {len(selected)}/{num_samples}...")
        
        else:
            raise ValueError(f"未知方法: {method}")
    
    # 复制选中的图像
    print(f"\n复制 {len(selected)} 张图像到 {output_dir}...")
    for i, img_path in enumerate(selected):
        dst_path = output_dir / f"seed_{i:04d}{img_path.suffix}"
        shutil.copy2(img_path, dst_path)
        
        if (i + 1) % 20 == 0:
            print(f"  已复制 {i+1}/{len(selected)}...")
    
    print(f"\n✓ 完成！已选择 {len(selected)} 张图像")
    print(f"  输出目录: {output_dir}")
    
    return selected


def create_preview_grid(image_dir, output_path, grid_size=(10, 10), thumb_size=(128, 128)):
    """
    创建图像网格预览
    """
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([f for f in image_dir.iterdir() 
                         if f.suffix.lower() in image_extensions])
    
    if not image_files:
        print("未找到图像")
        return
    
    rows, cols = grid_size
    max_images = min(rows * cols, len(image_files))
    
    print(f"创建 {rows}x{cols} 预览网格 ({max_images} 张图像)...")
    
    # 创建空白画布
    canvas = np.ones((rows * thumb_size[1], cols * thumb_size[0], 3), dtype=np.uint8) * 255
    
    for idx in range(max_images):
        img = cv2.imread(str(image_files[idx]))
        if img is None:
            continue
        
        # 调整大小
        img_resized = cv2.resize(img, thumb_size)
        
        # 计算位置
        row = idx // cols
        col = idx % cols
        y = row * thumb_size[1]
        x = col * thumb_size[0]
        
        # 放置图像
        canvas[y:y+thumb_size[1], x:x+thumb_size[0]] = img_resized
    
    # 保存
    cv2.imwrite(output_path, canvas)
    print(f"✓ 预览网格已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="精选种子数据集")
    parser.add_argument('input_dir', help='输入图像目录')
    parser.add_argument('--output', '-o', default='data/seed_dataset/images',
                       help='输出目录')
    parser.add_argument('--num', '-n', type=int, default=100,
                       help='选择数量')
    parser.add_argument('--method', '-m', choices=['uniform', 'random', 'histogram'],
                       default='uniform', help='选择方法')
    parser.add_argument('--preview', '-p', help='生成预览网格图')
    
    args = parser.parse_args()
    
    # 选择图像
    selected = select_diverse_frames(
        args.input_dir,
        args.output,
        args.num,
        args.method
    )
    
    # 生成预览
    if args.preview and selected:
        create_preview_grid(args.output, args.preview)
    
    print("\n下一步:")
    print(f"  1. 查看选中的图像: open {args.output}")
    print(f"  2. 使用LabelImg标注: labelImg {args.output}")
    print(f"  3. 标注文件保存到: {Path(args.output).parent / 'labels'}")


if __name__ == "__main__":
    main()
