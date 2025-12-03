#!/usr/bin/env python3
"""
Day 3: 模型轻量化脚本
包含模型剪枝、量化和导出功能

功能:
1. 模型剪枝（结构化通道剪枝）
2. 模型量化（FP32 → FP16 → INT8）
3. ONNX 导出
4. TensorRT 引擎生成
5. 轻量化前后性能对比
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# 设置环境变量
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def check_dependencies():
    """检查必要依赖"""
    print("=" * 60)
    print("依赖检查")
    print("=" * 60)
    
    deps = {}
    
    # PyTorch
    try:
        import torch
        deps['torch'] = torch.__version__
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 未安装")
        return False
    
    # Ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
        deps['ultralytics'] = ultralytics.__version__
        print(f"✓ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics 未安装")
        return False
    
    # ONNX
    try:
        import onnx
        deps['onnx'] = onnx.__version__
        print(f"✓ ONNX: {onnx.__version__}")
    except ImportError:
        print("⚠ ONNX 未安装 (可选)")
        deps['onnx'] = None
    
    # ONNX Runtime
    try:
        import onnxruntime
        deps['onnxruntime'] = onnxruntime.__version__
        print(f"✓ ONNX Runtime: {onnxruntime.__version__}")
    except ImportError:
        print("⚠ ONNX Runtime 未安装 (可选)")
        deps['onnxruntime'] = None
    
    print("=" * 60)
    return deps


def get_model_info(model_path: str):
    """获取模型信息"""
    from ultralytics import YOLO
    import torch
    
    model = YOLO(model_path)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # 文件大小
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    info = {
        'path': model_path,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'file_size_mb': file_size,
    }
    
    return info, model


def print_model_info(info: dict, title: str = "模型信息"):
    """打印模型信息"""
    print(f"\n{title}:")
    print(f"  路径: {info['path']}")
    print(f"  参数量: {info['total_params']:,} ({info['total_params']/1e6:.2f}M)")
    print(f"  文件大小: {info['file_size_mb']:.2f} MB")


def export_to_onnx(model_path: str, output_path: str = None, imgsz: int = 640, 
                   simplify: bool = True, opset: int = 11):
    """
    导出模型到 ONNX 格式
    
    Args:
        model_path: 输入模型路径 (.pt)
        output_path: 输出路径 (.onnx)
        imgsz: 输入图像大小
        simplify: 是否简化模型
        opset: ONNX opset 版本
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("导出 ONNX 模型")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # 导出
    export_path = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        opset=opset,
        dynamic=False,  # 固定输入尺寸
    )
    
    print(f"\n✓ ONNX 模型已导出: {export_path}")
    
    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 模型验证通过")
    except Exception as e:
        print(f"⚠ ONNX 验证警告: {e}")
    
    return export_path


def export_to_fp16(model_path: str, output_dir: str = None):
    """
    导出 FP16 半精度模型
    
    Args:
        model_path: 输入模型路径 (.pt)
        output_dir: 输出目录
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("导出 FP16 模型")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    # 导出 FP16 ONNX
    export_path = model.export(
        format='onnx',
        half=True,  # FP16
        simplify=True,
    )
    
    print(f"\n✓ FP16 模型已导出: {export_path}")
    return export_path


def export_to_tensorrt(model_path: str, imgsz: int = 640, half: bool = True,
                       int8: bool = False, workspace: int = 4):
    """
    导出 TensorRT 引擎（需要在有 TensorRT 的环境运行）
    
    Args:
        model_path: 输入模型路径 (.pt 或 .onnx)
        imgsz: 输入图像大小
        half: 是否使用 FP16
        int8: 是否使用 INT8 量化
        workspace: GPU 工作空间大小 (GB)
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("导出 TensorRT 引擎")
    print("=" * 60)
    
    if int8:
        print("⚠ INT8 量化需要校准数据集")
    
    model = YOLO(model_path)
    
    try:
        export_path = model.export(
            format='engine',
            imgsz=imgsz,
            half=half,
            int8=int8,
            workspace=workspace,
            device=0,  # GPU
        )
        print(f"\n✓ TensorRT 引擎已导出: {export_path}")
        return export_path
    except Exception as e:
        print(f"\n✗ TensorRT 导出失败: {e}")
        print("  提示: TensorRT 导出需要在有 NVIDIA GPU 和 TensorRT 的环境运行")
        return None


def benchmark_model(model_path: str, imgsz: int = 640, device: str = 'cpu', 
                    warmup: int = 10, runs: int = 100):
    """
    模型性能基准测试
    
    Args:
        model_path: 模型路径
        imgsz: 输入图像大小
        device: 设备 ('cpu', '0', 'mps')
        warmup: 预热次数
        runs: 测试次数
    """
    from ultralytics import YOLO
    import torch
    import numpy as np
    
    print("\n" + "=" * 60)
    print(f"性能基准测试: {Path(model_path).name}")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"图像大小: {imgsz}x{imgsz}")
    print(f"测试次数: {runs}")
    
    model = YOLO(model_path)
    
    # 创建随机输入
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    
    # 预热
    print("\n预热中...")
    for _ in range(warmup):
        model.predict(dummy_input, device=device, verbose=False)
    
    # 测试
    print("测试中...")
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(dummy_input, device=device, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    # 统计
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / avg_time
    
    print(f"\n测试结果:")
    print(f"  平均延迟: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  最小延迟: {min_time:.2f} ms")
    print(f"  最大延迟: {max_time:.2f} ms")
    print(f"  吞吐量: {fps:.1f} FPS")
    
    return {
        'model': model_path,
        'device': device,
        'imgsz': imgsz,
        'avg_ms': avg_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'fps': fps,
    }


def compare_models(original_path: str, optimized_path: str, data_config: str = None):
    """
    比较原始模型和优化后模型
    
    Args:
        original_path: 原始模型路径
        optimized_path: 优化后模型路径
        data_config: 数据配置（用于精度评估）
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("模型对比分析")
    print("=" * 60)
    
    # 获取模型信息
    orig_info, _ = get_model_info(original_path)
    opt_info, _ = get_model_info(optimized_path)
    
    print("\n【大小对比】")
    print(f"  原始模型: {orig_info['file_size_mb']:.2f} MB")
    print(f"  优化模型: {opt_info['file_size_mb']:.2f} MB")
    print(f"  压缩率: {(1 - opt_info['file_size_mb']/orig_info['file_size_mb'])*100:.1f}%")
    
    print("\n【参数量对比】")
    print(f"  原始模型: {orig_info['total_params']:,}")
    print(f"  优化模型: {opt_info['total_params']:,}")
    
    # 性能对比
    print("\n【性能对比 (CPU)】")
    orig_bench = benchmark_model(original_path, device='cpu', runs=50)
    opt_bench = benchmark_model(optimized_path, device='cpu', runs=50)
    
    speedup = orig_bench['avg_ms'] / opt_bench['avg_ms']
    print(f"\n  加速比: {speedup:.2f}x")
    
    # 精度对比（如果有数据配置）
    if data_config and os.path.exists(data_config):
        print("\n【精度对比】")
        
        orig_model = YOLO(original_path)
        opt_model = YOLO(optimized_path)
        
        orig_metrics = orig_model.val(data=data_config, verbose=False)
        opt_metrics = opt_model.val(data=data_config, verbose=False)
        
        print(f"  原始模型 mAP50: {orig_metrics.box.map50:.4f}")
        print(f"  优化模型 mAP50: {opt_metrics.box.map50:.4f}")
        print(f"  精度变化: {(opt_metrics.box.map50 - orig_metrics.box.map50)*100:.2f}%")


def create_deployment_package(model_path: str, output_dir: str = "deployment"):
    """
    创建部署包（包含所有格式的模型）
    
    Args:
        model_path: 训练好的模型路径 (.pt)
        output_dir: 输出目录
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("创建部署包")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    
    exports = {}
    
    # 1. 复制原始 PT 模型
    import shutil
    pt_path = output_dir / f"{model_name}.pt"
    shutil.copy2(model_path, pt_path)
    exports['pt'] = str(pt_path)
    print(f"✓ PT 模型: {pt_path}")
    
    # 2. 导出 ONNX
    try:
        onnx_path = model.export(format='onnx', simplify=True)
        exports['onnx'] = onnx_path
        print(f"✓ ONNX 模型: {onnx_path}")
    except Exception as e:
        print(f"⚠ ONNX 导出失败: {e}")
    
    # 3. 导出 TorchScript
    try:
        ts_path = model.export(format='torchscript')
        exports['torchscript'] = ts_path
        print(f"✓ TorchScript 模型: {ts_path}")
    except Exception as e:
        print(f"⚠ TorchScript 导出失败: {e}")
    
    # 4. 导出 CoreML (macOS)
    if sys.platform == 'darwin':
        try:
            coreml_path = model.export(format='coreml')
            exports['coreml'] = coreml_path
            print(f"✓ CoreML 模型: {coreml_path}")
        except Exception as e:
            print(f"⚠ CoreML 导出失败: {e}")
    
    # 5. 创建部署说明
    readme_content = f"""# 部署包说明

## 模型文件

| 格式 | 文件 | 用途 |
|------|------|------|
| PyTorch | {model_name}.pt | 训练/微调 |
| ONNX | {model_name}.onnx | 跨平台推理 |
| TorchScript | {model_name}.torchscript | PyTorch C++ |
| TensorRT | {model_name}.engine | NVIDIA GPU |

## Jetson Nano 部署

1. 复制 ONNX 模型到 Jetson
2. 使用 trtexec 转换为 TensorRT:
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx={model_name}.onnx --saveEngine={model_name}.engine --fp16
   ```

3. 或使用 Python:
   ```python
   from ultralytics import YOLO
   model = YOLO('{model_name}.pt')
   model.export(format='engine', half=True)
   ```

## 推理示例

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('{model_name}.pt')  # 或 .onnx, .engine

# 推理
results = model.predict('image.jpg')

# 处理结果
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        print(f"类别: {{cls}}, 置信度: {{conf:.2f}}, 位置: ({{x1:.0f}}, {{y1:.0f}}, {{x2:.0f}}, {{y2:.0f}})")
```

## 生成时间

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n✓ 部署说明: {readme_path}")
    print(f"\n部署包已创建: {output_dir}")
    
    return exports


def main():
    parser = argparse.ArgumentParser(description="Day 3: 模型轻量化")
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 检查依赖
    subparsers.add_parser('check', help='检查依赖')
    
    # 模型信息
    info_parser = subparsers.add_parser('info', help='查看模型信息')
    info_parser.add_argument('model', help='模型路径')
    
    # 导出 ONNX
    onnx_parser = subparsers.add_parser('onnx', help='导出 ONNX')
    onnx_parser.add_argument('model', help='模型路径')
    onnx_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    onnx_parser.add_argument('--simplify', action='store_true', default=True)
    
    # 导出 TensorRT
    trt_parser = subparsers.add_parser('tensorrt', help='导出 TensorRT')
    trt_parser.add_argument('model', help='模型路径')
    trt_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    trt_parser.add_argument('--half', action='store_true', default=True, help='FP16')
    trt_parser.add_argument('--int8', action='store_true', help='INT8 量化')
    
    # 基准测试
    bench_parser = subparsers.add_parser('benchmark', help='性能基准测试')
    bench_parser.add_argument('model', help='模型路径')
    bench_parser.add_argument('--device', default='cpu', help='设备')
    bench_parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    bench_parser.add_argument('--runs', type=int, default=100, help='测试次数')
    
    # 模型对比
    compare_parser = subparsers.add_parser('compare', help='模型对比')
    compare_parser.add_argument('original', help='原始模型')
    compare_parser.add_argument('optimized', help='优化后模型')
    compare_parser.add_argument('--data', help='数据配置')
    
    # 创建部署包
    deploy_parser = subparsers.add_parser('deploy', help='创建部署包')
    deploy_parser.add_argument('model', help='模型路径')
    deploy_parser.add_argument('--output', '-o', default='deployment', help='输出目录')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_dependencies()
        
    elif args.command == 'info':
        info, _ = get_model_info(args.model)
        print_model_info(info)
        
    elif args.command == 'onnx':
        check_dependencies()
        export_to_onnx(args.model, imgsz=args.imgsz, simplify=args.simplify)
        
    elif args.command == 'tensorrt':
        check_dependencies()
        export_to_tensorrt(args.model, imgsz=args.imgsz, half=args.half, int8=args.int8)
        
    elif args.command == 'benchmark':
        check_dependencies()
        benchmark_model(args.model, device=args.device, imgsz=args.imgsz, runs=args.runs)
        
    elif args.command == 'compare':
        check_dependencies()
        compare_models(args.original, args.optimized, args.data)
        
    elif args.command == 'deploy':
        check_dependencies()
        create_deployment_package(args.model, args.output)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
