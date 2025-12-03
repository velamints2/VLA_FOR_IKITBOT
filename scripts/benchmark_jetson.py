"""
Jetson Nano基准测试脚本
测试并记录Jetson Nano的基准性能
"""
import os
import sys
import time
import subprocess
import platform

try:
    import numpy as np
except ImportError:
    print("警告: NumPy未安装")
    np = None


def get_system_info():
    """获取系统信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    
    # 基本信息
    print(f"平台: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Jetson信息
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            print(f"Tegra版本: {f.read().strip()}")
    except:
        print("非Jetson平台或无法读取Tegra信息")
    
    # CUDA版本
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'release' in line:
                print(f"CUDA: {line.strip()}")
                break
    except:
        print("CUDA: 未安装或不可用")
    
    # 内存信息
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / 1024 / 1024
                    print(f"总内存: {mem_gb:.2f} GB")
                    break
    except:
        pass


def benchmark_cpu():
    """CPU性能测试"""
    print("\n" + "=" * 60)
    print("CPU基准测试")
    print("=" * 60)
    
    if np is None:
        print("跳过（NumPy未安装）")
        return
    
    # 矩阵乘法测试
    size = 1000
    print(f"测试: {size}x{size} 矩阵乘法...")
    
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    start = time.time()
    C = np.dot(A, B)
    elapsed = time.time() - start
    
    print(f"✓ 耗时: {elapsed:.3f} 秒")
    print(f"✓ 性能: {size**3 * 2 / elapsed / 1e9:.2f} GFLOPS")


def benchmark_gpu():
    """GPU性能测试"""
    print("\n" + "=" * 60)
    print("GPU基准测试")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("跳过（PyTorch未安装）")
        return
    
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    device = torch.device('cuda')
    print(f"设备: {torch.cuda.get_device_name(0)}")
    
    # 矩阵乘法测试
    size = 2000
    print(f"\n测试: {size}x{size} GPU矩阵乘法...")
    
    A = torch.rand(size, size).to(device)
    B = torch.rand(size, size).to(device)
    
    # 预热
    torch.cuda.synchronize()
    _ = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # 实际测试
    start = time.time()
    C = torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✓ 耗时: {elapsed:.3f} 秒")
    print(f"✓ 性能: {size**3 * 2 / elapsed / 1e9:.2f} GFLOPS")
    
    # 内存信息
    mem_allocated = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"\nGPU内存:")
    print(f"  已分配: {mem_allocated:.2f} MB")
    print(f"  已保留: {mem_reserved:.2f} MB")


def benchmark_camera():
    """摄像头性能测试"""
    print("\n" + "=" * 60)
    print("摄像头基准测试")
    print("=" * 60)
    
    try:
        import cv2
    except ImportError:
        print("跳过（OpenCV未安装）")
        return
    
    # 测试摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("测试: 读取100帧...")
    
    start = time.time()
    frame_count = 0
    
    for _ in range(100):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
    
    elapsed = time.time() - start
    fps = frame_count / elapsed
    
    cap.release()
    
    print(f"✓ 读取帧数: {frame_count}")
    print(f"✓ 平均FPS: {fps:.2f}")


def benchmark_inference():
    """推理性能测试（如果有模型）"""
    print("\n" + "=" * 60)
    print("推理性能基准测试")
    print("=" * 60)
    
    try:
        import torch
    except ImportError:
        print("跳过（PyTorch未安装）")
        return
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过")
        return
    
    # 创建简单的卷积网络测试
    print("测试: 简单CNN推理 (640x640输入)...")
    
    device = torch.device('cuda')
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1)
    ).to(device)
    
    model.eval()
    
    # 预热
    with torch.no_grad():
        dummy_input = torch.rand(1, 3, 640, 640).to(device)
        _ = model(dummy_input)
        torch.cuda.synchronize()
    
    # 测试100次推理
    num_runs = 100
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(dummy_input)
            torch.cuda.synchronize()
    
    elapsed = time.time() - start
    fps = num_runs / elapsed
    latency = elapsed / num_runs * 1000  # ms
    
    print(f"✓ 运行次数: {num_runs}")
    print(f"✓ 平均FPS: {fps:.2f}")
    print(f"✓ 平均延迟: {latency:.2f} ms")


def save_benchmark_report(output_path='docs/jetson_benchmark.txt'):
    """保存基准测试报告"""
    import datetime
    
    with open(output_path, 'w') as f:
        f.write("Jetson Nano 基准测试报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试时间: {datetime.datetime.now()}\n\n")
        
        # 这里可以添加更详细的报告内容
        f.write("注意: 详细结果请查看控制台输出\n")
    
    print(f"\n✓ 报告已保存到: {output_path}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Jetson Nano 基准测试工具")
    print("=" * 60)
    
    get_system_info()
    benchmark_cpu()
    benchmark_gpu()
    benchmark_camera()
    benchmark_inference()
    
    print("\n" + "=" * 60)
    print("基准测试完成")
    print("=" * 60)
    
    # 保存报告
    os.makedirs('docs', exist_ok=True)
    save_benchmark_report()


if __name__ == "__main__":
    main()
