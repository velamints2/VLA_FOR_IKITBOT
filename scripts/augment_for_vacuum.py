#!/usr/bin/env python3
"""
数据扩增管道：从开源数据集和现有数据生成高价值的扫地机器人训练库
主要技术：
- 透视变换（模拟低视角）
- 旋转/倾斜
- 亮度/对比度调整
- 添加扫地机器人常见的噪声
"""
import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
import argparse

class VacuumAugmenter:
    """扫地机器人视角数据增强器"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0
    
    def perspective_transform(self, img, intensity=0.3):
        """
        透视变换：模拟扫地机器人的低仰角视角
        intensity: 0-1, 值越大变换越剧烈
        """
        h, w = img.shape[:2]
        
        # 原始四个角
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # 目标透视（模拟低视角：上面压扁，下面拉宽）
        offset_x = int(w * intensity * 0.2)
        offset_y = int(h * intensity * 0.3)
        
        pts2 = np.float32([
            [offset_x, offset_y],                    # 左上
            [w - offset_x, offset_y],                # 右上
            [w, h],                                  # 右下
            [0, h]                                   # 左下
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (w, h))
        return result
    
    def rotate_and_tilt(self, img, angle_range=(-15, 15), tilt_range=(-10, 10)):
        """旋转和倾斜"""
        h, w = img.shape[:2]
        
        angle = random.uniform(*angle_range)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return result
    
    def lighting_adjustment(self, img, brightness_range=(-30, 30), contrast_range=(0.8, 1.2)):
        """亮度和对比度调整（模拟不同照明条件）"""
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        
        # 调整对比度和亮度
        result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_motion_blur(self, img, kernel_size=(5, 15)):
        """添加运动模糊（模拟移动中的拍摄）"""
        kernel_size = random.choice([5, 7, 9, 11])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 创建运动模糊核
        angle = random.uniform(0, 180)
        center = (kernel_size // 2, kernel_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        blur_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        blur_kernel = blur_kernel / blur_kernel.sum()
        
        result = cv2.filter2D(img, -1, blur_kernel)
        return result
    
    def add_gaussian_noise(self, img, noise_level=0.02):
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level * 255, img.shape)
        result = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        return result
    
    def add_sensor_artifacts(self, img):
        """添加传感器伪影（低质量摄像头效果）"""
        h, w = img.shape[:2]
        
        # 添加条纹噪声
        if random.random() > 0.5:
            stripe_frequency = random.randint(50, 150)
            for i in range(h):
                if i % stripe_frequency < 2:
                    img[i, :] = cv2.addWeighted(img[i, :], 0.9, img[i, :], 0.1, 20)
        
        # 添加暗角
        kernel_x = cv2.getGaussianKernel(w, w // 3)
        kernel_y = cv2.getGaussianKernel(h, h // 3)
        kernel = kernel_y @ kernel_x.T
        kernel = kernel / kernel.max()
        
        for c in range(3):
            img[:, :, c] = (img[:, :, c].astype(float) * kernel).astype(np.uint8)
        
        return img
    
    def augment_image(self, img, augment_type='full', intensity=0.5):
        """
        执行增强
        augment_type: 'perspective', 'rotate', 'lighting', 'full'
        """
        result = img.copy()
        
        if augment_type in ('perspective', 'full'):
            result = self.perspective_transform(result, intensity)
        
        if augment_type in ('rotate', 'full'):
            result = self.rotate_and_tilt(result)
        
        if augment_type in ('lighting', 'full'):
            result = self.lighting_adjustment(result)
        
        if augment_type == 'full':
            if random.random() > 0.5:
                result = self.add_motion_blur(result)
            if random.random() > 0.6:
                result = self.add_gaussian_noise(result, noise_level=0.01)
            if random.random() > 0.7:
                result = self.add_sensor_artifacts(result)
        
        return result
    
    def process_dataset(self, input_dir, multiplier=3, intensity=0.4):
        """
        处理整个数据集
        multiplier: 每张图像生成多少张增强图像
        """
        input_path = Path(input_dir)
        images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        if not images:
            print(f"❌ 在 {input_dir} 中找不到图像")
            return 0
        
        print(f"✓ 找到 {len(images)} 张图像")
        print(f"将生成 {len(images) * multiplier} 张增强图像\n")
        
        # 统计增强类型
        augment_stats = defaultdict(int)
        
        for img_idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠ 无法读取 {img_path}")
                continue
            
            base_name = img_path.stem
            
            # 原始图像
            output_path = self.output_dir / f"{base_name}_orig.jpg"
            cv2.imwrite(str(output_path), img)
            self.count += 1
            
            # 生成增强版本
            augment_types = ['perspective', 'rotate', 'lighting', 'full']
            selected_types = random.sample(augment_types, min(multiplier, len(augment_types)))
            
            for aug_idx, aug_type in enumerate(selected_types):
                intensity_val = random.uniform(intensity * 0.7, intensity * 1.3)
                aug_img = self.augment_image(img, aug_type, intensity_val)
                
                output_path = self.output_dir / f"{base_name}_aug{aug_idx + 1}_{aug_type}.jpg"
                cv2.imwrite(str(output_path), aug_img)
                augment_stats[aug_type] += 1
                self.count += 1
            
            # 进度
            if (img_idx + 1) % 50 == 0 or img_idx == len(images) - 1:
                pct = int(100 * (img_idx + 1) / len(images))
                print(f"  [{pct:3d}%] {img_idx + 1}/{len(images)} - 已生成 {self.count} 张图像")
        
        print(f"\n✓ 增强完成，共生成 {self.count} 张图像")
        print(f"\n增强类型分布：")
        for aug_type, count in sorted(augment_stats.items()):
            print(f"  {aug_type:15s}: {count:3d} 张")
        
        return self.count

def main():
    parser = argparse.ArgumentParser(description='扫地机器人视角数据增强')
    parser.add_argument('--input', default='data/seed_dataset_v2', help='输入图像目录')
    parser.add_argument('--output', default='data/augmented_dataset', help='输出目录')
    parser.add_argument('--multiplier', type=int, default=3, help='每张图像增强倍数')
    parser.add_argument('--intensity', type=float, default=0.4, help='增强强度 (0-1)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("扫地机器人视角数据增强管道")
    print("=" * 70 + "\n")
    
    augmenter = VacuumAugmenter(args.output)
    total = augmenter.process_dataset(args.input, args.multiplier, args.intensity)
    
    print("\n" + "=" * 70)
    print(f"✓ 生成高价值训练库完成")
    print(f"✓ 总数据量: {total} 张图像")
    print(f"✓ 输出目录: {args.output}")
    print("=" * 70)

if __name__ == '__main__':
    main()
