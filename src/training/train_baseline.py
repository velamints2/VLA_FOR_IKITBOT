#!/usr/bin/env python3
"""
Day 2: 基线模型训练脚本
支持 YOLOv8n, YOLOv10n, YOLO11n 等轻量级模型

功能:
1. 模型选择与配置
2. 快速验证训练（1-2 epoch）
3. 正式训练
4. 训练结果分析
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def check_environment():
    """检查训练环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # 检查 PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) 可用")
        else:
            print("⚠ GPU 不可用，将使用 CPU 训练（较慢）")
            
    except ImportError:
        print("✗ PyTorch 未安装")
        print("  安装命令: pip install torch torchvision")
        return False
    
    # 检查 ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✓ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics 未安装")
        print("  安装命令: pip install ultralytics")
        return False
    
    print("=" * 60)
    return True


def quick_validate(data_config: str, model_name: str = "yolov8n.pt"):
    """
    快速验证训练流程（1-2 epoch）
    用于验证数据集和配置是否正确
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("快速验证训练 (2 epochs)")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(model_name)
    
    # 快速训练
    results = model.train(
        data=data_config,
        epochs=2,
        imgsz=320,  # 小图像加速
        batch=4,
        device='mps' if sys.platform == 'darwin' else '0',
        project='runs/validate',
        name='quick_test',
        verbose=True
    )
    
    print("\n✓ 快速验证完成！数据集和配置正确。")
    return True


def train_baseline(
    data_config: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = 'auto',
    project: str = 'runs/train',
    name: str = None
):
    """
    训练基线模型
    
    Args:
        data_config: 数据配置文件路径
        model_name: 模型名称 (yolov8n.pt, yolov10n.pt, yolo11n.pt)
        epochs: 训练轮数
        imgsz: 输入图像大小
        batch: batch 大小
        device: 设备 ('0', 'cpu', 'mps', 'auto')
        project: 项目目录
        name: 实验名称
    """
    from ultralytics import YOLO
    import torch
    
    # 自动选择设备
    if device == 'auto':
        if torch.cuda.is_available():
            device = '0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # 生成实验名称
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.replace('.pt', '').replace('yolo', '')
        name = f"obstacle_{model_short}_{timestamp}"
    
    print("\n" + "=" * 60)
    print("基线模型训练")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"数据: {data_config}")
    print(f"轮数: {epochs}")
    print(f"图像大小: {imgsz}")
    print(f"Batch: {batch}")
    print(f"设备: {device}")
    print(f"实验名称: {name}")
    print("=" * 60)
    
    # 加载模型
    print("\n加载预训练模型...")
    model = YOLO(model_name)
    
    # 训练
    print("\n开始训练...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        
        # 训练参数
        patience=20,          # 早停耐心值
        save=True,            # 保存检查点
        save_period=10,       # 每10轮保存
        val=True,             # 训练时验证
        plots=True,           # 生成图表
        
        # 数据增强
        hsv_h=0.015,          # 色调增强
        hsv_s=0.7,            # 饱和度增强
        hsv_v=0.4,            # 明度增强
        degrees=10,           # 旋转角度
        translate=0.1,        # 平移
        scale=0.5,            # 缩放
        fliplr=0.5,           # 水平翻转
        mosaic=1.0,           # Mosaic 增强
        
        # 优化器
        optimizer='auto',
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率因子
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        verbose=True
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("✓ 训练完成！")
    print("=" * 60)
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
    print(f"最终模型: {results.save_dir}/weights/last.pt")
    print(f"训练日志: {results.save_dir}")
    
    return results


def evaluate_model(model_path: str, data_config: str):
    """评估模型性能"""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # 验证
    metrics = model.val(data=data_config)
    
    print(f"\n评估结果:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def analyze_results(results_dir: str):
    """分析训练结果"""
    results_dir = Path(results_dir)
    
    print("\n" + "=" * 60)
    print("训练结果分析")
    print("=" * 60)
    
    # 检查必要文件
    files = {
        'results.csv': '训练日志',
        'weights/best.pt': '最佳模型',
        'weights/last.pt': '最终模型',
        'confusion_matrix.png': '混淆矩阵',
        'results.png': '训练曲线',
        'val_batch0_pred.jpg': '验证预测示例'
    }
    
    print("\n生成的文件:")
    for f, desc in files.items():
        path = results_dir / f
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  ✓ {desc}: {f} ({size:.1f} KB)")
        else:
            print(f"  ✗ {desc}: {f} (未找到)")
    
    # 读取训练日志
    csv_path = results_dir / 'results.csv'
    if csv_path.exists():
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if rows:
            last_row = rows[-1]
            print(f"\n最终训练指标:")
            
            # 常见指标
            metrics = [
                ('train/box_loss', '训练Box损失'),
                ('train/cls_loss', '训练Cls损失'),
                ('val/box_loss', '验证Box损失'),
                ('val/cls_loss', '验证Cls损失'),
                ('metrics/mAP50(B)', 'mAP50'),
                ('metrics/mAP50-95(B)', 'mAP50-95'),
            ]
            
            for key, name in metrics:
                if key in last_row:
                    try:
                        val = float(last_row[key])
                        print(f"  {name}: {val:.4f}")
                    except:
                        pass


def main():
    parser = argparse.ArgumentParser(description="Day 2: 基线模型训练")
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 环境检查
    subparsers.add_parser('check', help='检查训练环境')
    
    # 快速验证
    validate_parser = subparsers.add_parser('validate', help='快速验证训练流程')
    validate_parser.add_argument('--data', default='configs/data.yaml', help='数据配置')
    validate_parser.add_argument('--model', default='yolov8n.pt', help='模型')
    
    # 训练
    train_parser = subparsers.add_parser('train', help='训练基线模型')
    train_parser.add_argument('--data', default='configs/data.yaml', help='数据配置')
    train_parser.add_argument('--model', default='yolov8n.pt', 
                             help='模型名称 (yolov8n.pt, yolov10n.pt, yolo11n.pt)')
    train_parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    train_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch大小')
    train_parser.add_argument('--device', default='auto', help='设备')
    train_parser.add_argument('--name', help='实验名称')
    
    # 评估
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('model', help='模型路径')
    eval_parser.add_argument('--data', default='configs/data.yaml', help='数据配置')
    
    # 分析
    analyze_parser = subparsers.add_parser('analyze', help='分析训练结果')
    analyze_parser.add_argument('results_dir', help='结果目录')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_environment()
        
    elif args.command == 'validate':
        if check_environment():
            quick_validate(args.data, args.model)
            
    elif args.command == 'train':
        if check_environment():
            train_baseline(
                data_config=args.data,
                model_name=args.model,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                name=args.name
            )
            
    elif args.command == 'eval':
        evaluate_model(args.model, args.data)
        
    elif args.command == 'analyze':
        analyze_results(args.results_dir)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
