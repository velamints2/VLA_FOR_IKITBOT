#!/usr/bin/env python3
"""
Day 2: 多GPU分布式训练脚本
支持 16x RTX 2080 GPU 服务器

功能:
1. 多GPU数据并行训练 (DDP)
2. 自动GPU检测与分配
3. 混合精度训练 (AMP)
4. 梯度累积支持
5. 训练监控与日志

使用方法:
    # 单机多卡 (推荐)
    python train_distributed.py ddp --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    
    # 自动使用所有可用GPU
    python train_distributed.py ddp --gpus all
    
    # 指定部分GPU
    python train_distributed.py ddp --gpus 0,1,2,3
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional


def check_multi_gpu_environment():
    """检查多GPU训练环境"""
    print("=" * 70)
    print("多GPU训练环境检查 (16x RTX 2080 Server)")
    print("=" * 70)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA: {torch.version.cuda}")
        
        if not torch.cuda.is_available():
            print("✗ CUDA 不可用")
            return False, []
        
        # 检测所有GPU
        gpu_count = torch.cuda.device_count()
        print(f"\n发现 {gpu_count} 个 GPU:")
        
        gpu_info = []
        total_memory = 0
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            total_memory += mem_gb
            gpu_info.append({
                'id': i,
                'name': props.name,
                'memory': mem_gb,
                'compute': f"{props.major}.{props.minor}"
            })
            print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB, SM {props.major}.{props.minor})")
        
        print(f"\n总显存: {total_memory:.1f} GB")
        
        # 检查 NCCL 后端
        if torch.distributed.is_nccl_available():
            print("✓ NCCL 后端可用 (推荐用于多GPU)")
        else:
            print("⚠ NCCL 不可用，将使用 Gloo 后端")
        
        # 检查 ultralytics
        from ultralytics import YOLO
        import ultralytics
        print(f"✓ Ultralytics: {ultralytics.__version__}")
        
        print("=" * 70)
        return True, gpu_info
        
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        return False, []


def get_optimal_batch_size(gpu_count: int, gpu_memory_gb: float, imgsz: int) -> int:
    """
    根据GPU数量和显存计算最优batch size
    
    RTX 2080 有 8GB 显存
    YOLOv8n 在 640x640 下约需 2-3GB 显存
    """
    # 单卡安全batch size估算
    if imgsz <= 320:
        single_gpu_batch = 32
    elif imgsz <= 480:
        single_gpu_batch = 24
    elif imgsz <= 640:
        single_gpu_batch = 16
    else:
        single_gpu_batch = 8
    
    # RTX 2080 8GB 显存调整
    if gpu_memory_gb < 10:
        single_gpu_batch = int(single_gpu_batch * 0.8)
    
    # 总batch size = 单卡batch * GPU数量
    total_batch = single_gpu_batch * gpu_count
    
    return total_batch


def train_ddp(
    data_config: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,  # -1 表示自动计算
    gpus: str = "all",
    project: str = "runs/train",
    name: str = None,
    workers: int = 8,
    amp: bool = True,
    cache: bool = True,
    resume: str = None
):
    """
    使用 DDP (Distributed Data Parallel) 进行多GPU训练
    
    Args:
        data_config: 数据配置文件
        model_name: 模型名称
        epochs: 训练轮数
        imgsz: 输入图像大小
        batch: batch大小 (-1 自动计算)
        gpus: GPU列表 ("all" 或 "0,1,2,3")
        project: 项目目录
        name: 实验名称
        workers: 数据加载线程数
        amp: 是否使用混合精度
        cache: 是否缓存图像到内存
        resume: 恢复训练的检查点路径
    """
    import torch
    from ultralytics import YOLO
    
    # 解析GPU列表
    if gpus == "all":
        gpu_count = torch.cuda.device_count()
        gpu_list = list(range(gpu_count))
    else:
        gpu_list = [int(g.strip()) for g in gpus.split(",")]
        gpu_count = len(gpu_list)
    
    # 设置CUDA可见设备
    device_str = ",".join(map(str, gpu_list))
    
    # 自动计算batch size
    if batch == -1:
        props = torch.cuda.get_device_properties(gpu_list[0])
        gpu_memory = props.total_memory / 1024**3
        batch = get_optimal_batch_size(gpu_count, gpu_memory, imgsz)
    
    # 计算每卡batch
    batch_per_gpu = batch // gpu_count
    effective_batch = batch_per_gpu * gpu_count
    
    # 生成实验名称
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_name.replace('.pt', '').replace('yolo', '')
        name = f"obstacle_{model_short}_{gpu_count}gpu_{timestamp}"
    
    print("\n" + "=" * 70)
    print("多GPU分布式训练 (DDP)")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"数据: {data_config}")
    print(f"GPU数量: {gpu_count} (设备: {device_str})")
    print(f"Batch Size: {effective_batch} (每卡 {batch_per_gpu})")
    print(f"图像大小: {imgsz}")
    print(f"轮数: {epochs}")
    print(f"混合精度: {'启用' if amp else '禁用'}")
    print(f"数据缓存: {'启用' if cache else '禁用'}")
    print(f"数据加载线程: {workers}")
    print(f"实验名称: {name}")
    print("=" * 70)
    
    # 加载模型
    print("\n加载预训练模型...")
    if resume:
        model = YOLO(resume)
        print(f"从检查点恢复: {resume}")
    else:
        model = YOLO(model_name)
    
    # 训练配置
    train_args = {
        'data': data_config,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': effective_batch,
        'device': gpu_list,  # 传入GPU列表启用DDP
        'project': project,
        'name': name,
        'workers': workers,
        'amp': amp,
        'cache': cache,
        
        # 训练参数
        'patience': 30,
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,  # 多GPU时可以用更强的增强
        
        # 优化器
        'optimizer': 'AdamW',  # AdamW 在大batch下更稳定
        'lr0': 0.001 * (effective_batch / 64),  # 线性缩放学习率
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        
        'verbose': True,
        'exist_ok': True,
    }
    
    # 恢复训练
    if resume:
        train_args['resume'] = True
    
    # 开始训练
    print("\n" + "=" * 70)
    print("🚀 开始多GPU训练...")
    print("=" * 70)
    
    results = model.train(**train_args)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("✓ 多GPU训练完成！")
    print("=" * 70)
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
    print(f"最终模型: {results.save_dir}/weights/last.pt")
    print(f"训练日志: {results.save_dir}")
    
    # 计算训练统计
    if hasattr(results, 'results_dict'):
        print(f"\n最终指标:")
        for k, v in results.results_dict.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    
    return results


def estimate_training_time(
    epochs: int,
    dataset_size: int,
    batch_size: int,
    gpu_count: int,
    imgsz: int = 640
):
    """估算训练时间"""
    # 基于RTX 2080的经验数据
    # YOLOv8n 在 640x640, batch=16 时约 0.05s/image
    
    base_time_per_image = 0.05  # 秒
    
    # 根据图像大小调整
    size_factor = (imgsz / 640) ** 2
    
    # 多GPU加速（考虑通信开销，效率约85%）
    gpu_efficiency = 0.85 if gpu_count > 1 else 1.0
    effective_speedup = gpu_count * gpu_efficiency
    
    # 计算每个epoch时间
    images_per_epoch = dataset_size
    time_per_epoch = (images_per_epoch * base_time_per_image * size_factor) / effective_speedup
    
    total_time = time_per_epoch * epochs
    
    return {
        'time_per_epoch_min': time_per_epoch / 60,
        'total_time_hours': total_time / 3600,
        'gpu_efficiency': gpu_efficiency,
        'effective_speedup': effective_speedup
    }


def benchmark_multi_gpu(gpus: str = "all"):
    """多GPU性能基准测试"""
    import torch
    from ultralytics import YOLO
    import time
    
    print("\n" + "=" * 70)
    print("多GPU性能基准测试")
    print("=" * 70)
    
    # 解析GPU
    if gpus == "all":
        gpu_count = torch.cuda.device_count()
        gpu_list = list(range(gpu_count))
    else:
        gpu_list = [int(g.strip()) for g in gpus.split(",")]
        gpu_count = len(gpu_list)
    
    print(f"测试GPU: {gpu_list}")
    
    # 创建测试数据
    model = YOLO("yolov8n.pt")
    
    # 测试不同GPU配置
    configs = [
        [0],  # 单卡
        gpu_list[:2] if gpu_count >= 2 else None,  # 2卡
        gpu_list[:4] if gpu_count >= 4 else None,  # 4卡
        gpu_list[:8] if gpu_count >= 8 else None,  # 8卡
        gpu_list if gpu_count > 8 else None,  # 全部
    ]
    
    print(f"\n{'GPU配置':<15} {'Batch':<10} {'吞吐量':<15} {'加速比':<10}")
    print("-" * 50)
    
    baseline_throughput = None
    
    for config in configs:
        if config is None:
            continue
        
        n_gpus = len(config)
        batch_size = 16 * n_gpus
        
        # 测试推理吞吐量
        dummy_input = torch.randn(batch_size, 3, 640, 640)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.model(dummy_input.cuda(config[0]))
        
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            with torch.no_grad():
                _ = model.model(dummy_input.cuda(config[0]))
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (batch_size * iterations) / elapsed
        
        if baseline_throughput is None:
            baseline_throughput = throughput
            speedup = 1.0
        else:
            speedup = throughput / baseline_throughput
        
        print(f"{n_gpus} GPU{'s' if n_gpus > 1 else '':<8} {batch_size:<10} {throughput:.1f} img/s{'':<5} {speedup:.2f}x")
    
    print("-" * 50)


def create_slurm_script(
    job_name: str,
    gpus: int = 16,
    epochs: int = 100,
    model: str = "yolov8n.pt"
):
    """生成SLURM集群提交脚本"""
    
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --cpus-per-task={gpus * 4}
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 加载环境
source ~/.bashrc
conda activate obstacle_detection

# 设置分布式训练环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE={gpus}
export NCCL_DEBUG=INFO

# 运行训练
cd $SLURM_SUBMIT_DIR

python src/training/train_distributed.py ddp \\
    --data configs/data.yaml \\
    --model {model} \\
    --epochs {epochs} \\
    --gpus all \\
    --amp \\
    --cache

echo "训练完成!"
"""
    
    script_path = Path("scripts/slurm_train.sh")
    script_path.parent.mkdir(exist_ok=True)
    script_path.write_text(script)
    print(f"✓ SLURM脚本已生成: {script_path}")
    print(f"  提交命令: sbatch {script_path}")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Day 2: 多GPU分布式训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 检查多GPU环境
    python train_distributed.py check
    
    # 使用所有GPU训练
    python train_distributed.py ddp --gpus all --epochs 100
    
    # 使用指定GPU训练
    python train_distributed.py ddp --gpus 0,1,2,3,4,5,6,7 --epochs 100
    
    # 估算训练时间
    python train_distributed.py estimate --dataset-size 1000 --epochs 100 --gpus 16
    
    # 生成SLURM提交脚本
    python train_distributed.py slurm --gpus 16 --epochs 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 环境检查
    check_parser = subparsers.add_parser('check', help='检查多GPU环境')
    
    # DDP训练
    ddp_parser = subparsers.add_parser('ddp', help='多GPU DDP训练')
    ddp_parser.add_argument('--data', default='configs/data.yaml', help='数据配置')
    ddp_parser.add_argument('--model', default='yolov8n.pt', help='模型名称')
    ddp_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    ddp_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    ddp_parser.add_argument('--batch', type=int, default=-1, help='Batch大小 (-1自动)')
    ddp_parser.add_argument('--gpus', default='all', help='GPU列表 (all 或 0,1,2,3)')
    ddp_parser.add_argument('--workers', type=int, default=8, help='数据加载线程')
    ddp_parser.add_argument('--amp', action='store_true', default=True, help='混合精度')
    ddp_parser.add_argument('--no-amp', dest='amp', action='store_false', help='禁用混合精度')
    ddp_parser.add_argument('--cache', action='store_true', default=True, help='缓存数据')
    ddp_parser.add_argument('--no-cache', dest='cache', action='store_false', help='不缓存')
    ddp_parser.add_argument('--resume', help='恢复训练的检查点')
    ddp_parser.add_argument('--name', help='实验名称')
    
    # 基准测试
    bench_parser = subparsers.add_parser('benchmark', help='多GPU基准测试')
    bench_parser.add_argument('--gpus', default='all', help='GPU列表')
    
    # 时间估算
    est_parser = subparsers.add_parser('estimate', help='估算训练时间')
    est_parser.add_argument('--dataset-size', type=int, default=1000, help='数据集大小')
    est_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    est_parser.add_argument('--batch', type=int, default=256, help='Batch大小')
    est_parser.add_argument('--gpus', type=int, default=16, help='GPU数量')
    est_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    
    # SLURM脚本
    slurm_parser = subparsers.add_parser('slurm', help='生成SLURM提交脚本')
    slurm_parser.add_argument('--name', default='obstacle_train', help='作业名称')
    slurm_parser.add_argument('--gpus', type=int, default=16, help='GPU数量')
    slurm_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    slurm_parser.add_argument('--model', default='yolov8n.pt', help='模型')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_multi_gpu_environment()
        
    elif args.command == 'ddp':
        ok, _ = check_multi_gpu_environment()
        if ok:
            train_ddp(
                data_config=args.data,
                model_name=args.model,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                gpus=args.gpus,
                workers=args.workers,
                amp=args.amp,
                cache=args.cache,
                resume=args.resume,
                name=args.name
            )
        else:
            print("错误: 多GPU环境检查失败")
            sys.exit(1)
            
    elif args.command == 'benchmark':
        ok, _ = check_multi_gpu_environment()
        if ok:
            benchmark_multi_gpu(args.gpus)
            
    elif args.command == 'estimate':
        est = estimate_training_time(
            epochs=args.epochs,
            dataset_size=args.dataset_size,
            batch_size=args.batch,
            gpu_count=args.gpus,
            imgsz=args.imgsz
        )
        
        print("\n" + "=" * 50)
        print("训练时间估算")
        print("=" * 50)
        print(f"数据集大小: {args.dataset_size} 张")
        print(f"训练轮数: {args.epochs}")
        print(f"GPU数量: {args.gpus}")
        print(f"Batch大小: {args.batch}")
        print("-" * 50)
        print(f"GPU效率: {est['gpu_efficiency']*100:.0f}%")
        print(f"有效加速: {est['effective_speedup']:.1f}x")
        print(f"每轮时间: {est['time_per_epoch_min']:.1f} 分钟")
        print(f"总时间: {est['total_time_hours']:.1f} 小时")
        print("=" * 50)
        
    elif args.command == 'slurm':
        create_slurm_script(
            job_name=args.name,
            gpus=args.gpus,
            epochs=args.epochs,
            model=args.model
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
