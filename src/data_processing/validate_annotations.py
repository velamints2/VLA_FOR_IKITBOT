"""
验证标注质量
检查标注文件是否正确，与图像一一对应
"""
import os
import sys
import argparse
from pathlib import Path


def validate_annotations(image_dir, label_dir, class_file=None):
    """验证标注文件"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    print("=" * 60)
    print("标注验证")
    print("=" * 60)
    
    # 收集图像和标签
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = {f.stem: f for f in image_dir.iterdir() 
             if f.suffix.lower() in image_extensions}
    labels = {f.stem: f for f in label_dir.iterdir() 
             if f.suffix == '.txt'}
    
    print(f"图像数量: {len(images)}")
    print(f"标签数量: {len(labels)}")
    
    # 检查一致性
    missing_labels = set(images.keys()) - set(labels.keys())
    extra_labels = set(labels.keys()) - set(images.keys())
    matched = set(images.keys()) & set(labels.keys())
    
    print(f"\n匹配数量: {len(matched)}")
    
    if missing_labels:
        print(f"\n⚠️  缺少标签的图像 ({len(missing_labels)}):")
        for name in sorted(list(missing_labels)[:10]):
            print(f"  - {name}")
        if len(missing_labels) > 10:
            print(f"  ... 还有 {len(missing_labels)-10} 个")
    
    if extra_labels:
        print(f"\n⚠️  多余的标签文件 ({len(extra_labels)}):")
        for name in sorted(list(extra_labels)[:10]):
            print(f"  - {name}")
        if len(extra_labels) > 10:
            print(f"  ... 还有 {len(extra_labels)-10} 个")
    
    # 验证标签内容
    if matched:
        print(f"\n验证标签格式...")
        valid_count = 0
        invalid_files = []
        empty_files = []
        
        class_counts = {}
        total_boxes = 0
        
        for name in matched:
            label_path = labels[name]
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    empty_files.append(name)
                    continue
                
                file_valid = True
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        invalid_files.append((name, f"格式错误: {line.strip()}"))
                        file_valid = False
                        break
                    
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        # 检查范围
                        if not all(0 <= c <= 1 for c in coords):
                            invalid_files.append((name, f"坐标超出范围: {line.strip()}"))
                            file_valid = False
                            break
                        
                        # 统计类别
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_boxes += 1
                        
                    except ValueError:
                        invalid_files.append((name, f"数值错误: {line.strip()}"))
                        file_valid = False
                        break
                
                if file_valid:
                    valid_count += 1
                    
            except Exception as e:
                invalid_files.append((name, str(e)))
        
        print(f"\n有效标签: {valid_count}/{len(matched)}")
        
        if empty_files:
            print(f"\n⚠️  空标签文件 ({len(empty_files)}):")
            for name in empty_files[:5]:
                print(f"  - {name}")
            if len(empty_files) > 5:
                print(f"  ... 还有 {len(empty_files)-5} 个")
        
        if invalid_files:
            print(f"\n❌ 无效标签文件 ({len(invalid_files)}):")
            for name, error in invalid_files[:5]:
                print(f"  - {name}: {error}")
            if len(invalid_files) > 5:
                print(f"  ... 还有 {len(invalid_files)-5} 个")
        
        if class_counts:
            print(f"\n类别统计:")
            print(f"总标注框数: {total_boxes}")
            for class_id, count in sorted(class_counts.items()):
                print(f"  类别 {class_id}: {count} 个 ({count/total_boxes*100:.1f}%)")
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    if not missing_labels and not extra_labels and not invalid_files:
        print("✓ 所有检查通过！")
        print(f"  - {len(matched)} 个有效样本")
        print(f"  - {total_boxes} 个标注框")
        return True
    else:
        print("⚠️  发现问题:")
        if missing_labels:
            print(f"  - {len(missing_labels)} 个图像缺少标签")
        if extra_labels:
            print(f"  - {len(extra_labels)} 个多余标签")
        if invalid_files:
            print(f"  - {len(invalid_files)} 个无效标签")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证标注质量")
    parser.add_argument('--images', required=True, help='图像目录')
    parser.add_argument('--labels', required=True, help='标签目录')
    parser.add_argument('--classes', help='类别文件')
    
    args = parser.parse_args()
    
    success = validate_annotations(args.images, args.labels, args.classes)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
