#!/usr/bin/env python3
"""
Day 3: Jetson Nano 功能完整度测试脚本

测试内容:
1. 系统信息检测
2. CUDA/cuDNN/TensorRT 环境
3. 摄像头功能
4. Python 依赖
5. 推理性能基准
6. 内存和存储状态

使用方法:
    python jetson_test.py all        # 运行所有测试
    python jetson_test.py system     # 系统信息
    python jetson_test.py cuda       # CUDA环境
    python jetson_test.py camera     # 摄像头测试
    python jetson_test.py inference  # 推理测试
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_cmd(cmd: str, timeout: int = 30) -> tuple:
    """执行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "命令超时"
    except Exception as e:
        return False, "", str(e)


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, success: bool, detail: str = ""):
    """打印测试结果"""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {name}: {detail}")


class JetsonTester:
    """Jetson Nano 功能测试器"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
    
    def test_system_info(self):
        """测试系统信息"""
        print_header("系统信息检测")
        
        # Jetson 型号
        ok, out, _ = run_cmd("cat /proc/device-tree/model 2>/dev/null || echo 'Unknown'")
        print_result("设备型号", ok, out)
        self.results['model'] = out
        
        # L4T 版本
        ok, out, _ = run_cmd("head -n 1 /etc/nv_tegra_release 2>/dev/null || echo 'N/A'")
        print_result("L4T 版本", ok, out[:60] if out else "N/A")
        self.results['l4t'] = out
        
        # JetPack 版本
        ok, out, _ = run_cmd("apt-cache show nvidia-jetpack 2>/dev/null | grep Version | head -1")
        version = out.split(":")[1].strip() if ":" in out else "N/A"
        print_result("JetPack 版本", ok, version)
        self.results['jetpack'] = version
        
        # 内核版本
        ok, out, _ = run_cmd("uname -r")
        print_result("内核版本", ok, out)
        
        # Ubuntu 版本
        ok, out, _ = run_cmd("lsb_release -d 2>/dev/null | cut -f2")
        print_result("Ubuntu 版本", ok, out)
        
        # 主机名
        ok, out, _ = run_cmd("hostname")
        print_result("主机名", ok, out)
        
        # CPU 信息
        ok, out, _ = run_cmd("lscpu | grep 'Model name' | cut -d: -f2")
        print_result("CPU", ok, out.strip() if out else "ARM Cortex-A57")
        
        # CPU 核心数
        ok, out, _ = run_cmd("nproc")
        print_result("CPU 核心", ok, f"{out} 核")
        
        return True
    
    def test_memory_storage(self):
        """测试内存和存储"""
        print_header("内存与存储")
        
        # 内存
        ok, out, _ = run_cmd("free -h | grep Mem | awk '{print $2}'")
        print_result("总内存", ok, out)
        self.results['memory'] = out
        
        ok, out, _ = run_cmd("free -h | grep Mem | awk '{print $3}'")
        print_result("已用内存", ok, out)
        
        ok, out, _ = run_cmd("free -h | grep Mem | awk '{print $7}'")
        print_result("可用内存", ok, out)
        
        # Swap
        ok, out, _ = run_cmd("free -h | grep Swap | awk '{print $2}'")
        print_result("Swap 大小", ok, out if out else "0B")
        
        # 磁盘
        ok, out, _ = run_cmd("df -h / | tail -1 | awk '{print $2}'")
        print_result("磁盘总量", ok, out)
        
        ok, out, _ = run_cmd("df -h / | tail -1 | awk '{print $4}'")
        print_result("磁盘可用", ok, out)
        self.results['disk_free'] = out
        
        ok, out, _ = run_cmd("df -h / | tail -1 | awk '{print $5}'")
        print_result("磁盘使用率", ok, out)
        
        return True
    
    def test_cuda_environment(self):
        """测试 CUDA 环境"""
        print_header("CUDA 环境检测")
        
        # CUDA 版本
        ok, out, _ = run_cmd("nvcc --version 2>/dev/null | grep release | awk '{print $6}'")
        cuda_version = out.replace(",", "") if out else "N/A"
        print_result("CUDA 版本", ok and out, cuda_version)
        self.results['cuda'] = cuda_version
        
        # cuDNN 版本
        ok, out, _ = run_cmd("cat /usr/include/cudnn_version.h 2>/dev/null | grep CUDNN_MAJOR -A 2 | head -3")
        if ok and out:
            lines = out.split('\n')
            major = minor = patch = "?"
            for line in lines:
                if 'CUDNN_MAJOR' in line:
                    major = line.split()[-1]
                elif 'CUDNN_MINOR' in line:
                    minor = line.split()[-1]
                elif 'CUDNN_PATCHLEVEL' in line:
                    patch = line.split()[-1]
            cudnn_version = f"{major}.{minor}.{patch}"
        else:
            # 备用方法
            ok, out, _ = run_cmd("dpkg -l | grep cudnn | head -1 | awk '{print $3}'")
            cudnn_version = out if out else "N/A"
        print_result("cuDNN 版本", cudnn_version != "N/A", cudnn_version)
        self.results['cudnn'] = cudnn_version
        
        # TensorRT 版本
        ok, out, _ = run_cmd("dpkg -l | grep tensorrt | grep ii | head -1 | awk '{print $3}'")
        trt_version = out.split("-")[0] if out else "N/A"
        print_result("TensorRT 版本", ok and out, trt_version)
        self.results['tensorrt'] = trt_version
        
        # GPU 信息
        ok, out, _ = run_cmd("tegrastats --interval 100 --stop 2>/dev/null & sleep 0.2 && kill %1 2>/dev/null; cat /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq 2>/dev/null")
        if not ok or not out:
            ok, out, _ = run_cmd("cat /sys/kernel/debug/clk/gpc/clk_rate 2>/dev/null || echo '未知'")
        gpu_freq = f"{int(out)//1000000} MHz" if out.isdigit() else out
        print_result("GPU 频率", True, gpu_freq)
        
        # 检查 nvidia-smi (Jetson 上可能没有)
        ok, out, _ = run_cmd("which nvidia-smi")
        if ok:
            ok2, out2, _ = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null")
            print_result("nvidia-smi", ok2, out2 if out2 else "可用但无输出")
        else:
            print_result("nvidia-smi", False, "Jetson 使用 tegrastats")
        
        return cuda_version != "N/A"
    
    def test_python_environment(self):
        """测试 Python 环境"""
        print_header("Python 环境检测")
        
        # Python 版本
        ok, out, _ = run_cmd("python3 --version")
        print_result("Python 版本", ok, out)
        self.results['python'] = out
        
        # pip 版本
        ok, out, _ = run_cmd("pip3 --version | awk '{print $2}'")
        print_result("pip 版本", ok, out)
        
        # 关键包检测
        packages = [
            ('numpy', 'numpy'),
            ('opencv-python', 'cv2'),
            ('torch', 'torch'),
            ('torchvision', 'torchvision'),
            ('tensorrt', 'tensorrt'),
            ('pycuda', 'pycuda'),
            ('onnx', 'onnx'),
            ('onnxruntime', 'onnxruntime'),
            ('pillow', 'PIL'),
        ]
        
        print("\n关键 Python 包:")
        for pkg_name, import_name in packages:
            ok, out, _ = run_cmd(f"python3 -c \"import {import_name}; print({import_name}.__version__)\" 2>/dev/null")
            if ok and out:
                print_result(f"  {pkg_name}", True, out)
                self.results[f'pkg_{pkg_name}'] = out
            else:
                print_result(f"  {pkg_name}", False, "未安装")
                self.results[f'pkg_{pkg_name}'] = None
        
        return True
    
    def test_camera(self):
        """测试摄像头"""
        print_header("摄像头检测")
        
        # 检查 video 设备
        ok, out, _ = run_cmd("ls /dev/video* 2>/dev/null")
        if ok and out:
            devices = out.split('\n')
            print_result("视频设备", True, f"发现 {len(devices)} 个设备")
            for dev in devices:
                print(f"    - {dev}")
            self.results['cameras'] = devices
        else:
            print_result("视频设备", False, "未发现摄像头")
            self.results['cameras'] = []
        
        # 检查 CSI 摄像头
        ok, out, _ = run_cmd("ls /dev/video0 2>/dev/null")
        csi_available = ok
        print_result("CSI 摄像头 (video0)", csi_available, "可用" if csi_available else "未检测到")
        
        # 检查 USB 摄像头
        ok, out, _ = run_cmd("lsusb | grep -i camera")
        if ok and out:
            print_result("USB 摄像头", True, "已连接")
            print(f"    {out}")
        else:
            ok, out, _ = run_cmd("lsusb | grep -i webcam")
            if ok and out:
                print_result("USB 摄像头", True, "已连接")
            else:
                print_result("USB 摄像头", False, "未检测到")
        
        # 检查 RealSense
        ok, out, _ = run_cmd("lsusb | grep -i 'Intel.*RealSense'")
        if ok and out:
            print_result("RealSense", True, "已连接")
            self.results['realsense'] = True
        else:
            print_result("RealSense", False, "未检测到")
            self.results['realsense'] = False
        
        # 尝试用 OpenCV 读取
        ok, out, _ = run_cmd("""python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f'{frame.shape[1]}x{frame.shape[0]}')
else:
    print('FAIL')
cap.release()
" 2>/dev/null""")
        if ok and out and out != 'FAIL':
            print_result("OpenCV 摄像头测试", True, f"分辨率: {out}")
        else:
            print_result("OpenCV 摄像头测试", False, "无法读取")
        
        return len(self.results.get('cameras', [])) > 0
    
    def test_inference_performance(self):
        """测试推理性能"""
        print_header("推理性能基准测试")
        
        # 检查是否有预训练模型
        test_script = '''
import time
import numpy as np

try:
    import torch
    
    # 检查 CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA: {cuda_available}")
    
    if cuda_available:
        device = torch.device("cuda")
        
        # 创建测试张量
        x = torch.randn(1, 3, 640, 640).to(device)
        
        # 简单卷积测试
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        
        # Warmup
        for _ in range(5):
            _ = conv(x)
        torch.cuda.synchronize()
        
        # 计时
        times = []
        for _ in range(20):
            start = time.time()
            _ = conv(x)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
        
        avg = np.mean(times)
        std = np.std(times)
        print(f"Conv2d 640x640: {avg:.2f} ± {std:.2f} ms")
    else:
        print("CUDA 不可用，跳过 GPU 测试")
        
except ImportError as e:
    print(f"缺少依赖: {e}")
except Exception as e:
    print(f"错误: {e}")
'''
        
        ok, out, err = run_cmd(f'python3 -c "{test_script}"', timeout=60)
        if ok:
            for line in out.split('\n'):
                if line.strip():
                    if 'CUDA' in line:
                        cuda_ok = 'True' in line
                        print_result("CUDA 可用", cuda_ok, line.split(': ')[1] if ': ' in line else line)
                    elif 'Conv2d' in line:
                        print_result("GPU 推理测试", True, line)
                    elif '错误' in line or '缺少' in line:
                        print_result("测试", False, line)
        else:
            print_result("推理测试", False, err or "执行失败")
        
        # TensorRT 推理测试
        trt_test = '''
try:
    import tensorrt as trt
    print(f"TensorRT {trt.__version__}")
    
    # 创建 logger 和 builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print("TensorRT Builder 创建成功")
except ImportError:
    print("TensorRT 未安装")
except Exception as e:
    print(f"TensorRT 错误: {e}")
'''
        
        ok, out, _ = run_cmd(f'python3 -c "{trt_test}"', timeout=30)
        if ok and 'TensorRT' in out and '错误' not in out:
            for line in out.split('\n'):
                if line.strip():
                    print_result("TensorRT", True, line)
        else:
            print_result("TensorRT", False, out or "不可用")
        
        return True
    
    def test_network(self):
        """测试网络连接"""
        print_header("网络状态")
        
        # IP 地址
        ok, out, _ = run_cmd("hostname -I | awk '{print $1}'")
        print_result("IP 地址", ok, out)
        
        # 网络接口
        ok, out, _ = run_cmd("ip link show | grep 'state UP' | awk -F: '{print $2}'")
        interfaces = [i.strip() for i in out.split('\n') if i.strip()] if out else []
        print_result("活动网卡", len(interfaces) > 0, ', '.join(interfaces) if interfaces else "无")
        
        # 测试外网连接
        ok, out, _ = run_cmd("ping -c 1 -W 2 8.8.8.8 2>/dev/null && echo 'OK' || echo 'FAIL'")
        print_result("外网连接", 'OK' in out, "正常" if 'OK' in out else "无法连接")
        
        # SSH 服务
        ok, out, _ = run_cmd("systemctl is-active ssh 2>/dev/null || service ssh status 2>/dev/null | grep running")
        ssh_ok = ok or 'running' in str(out).lower()
        print_result("SSH 服务", ssh_ok, "运行中" if ssh_ok else "未运行")
        
        return True
    
    def test_power_mode(self):
        """测试电源模式"""
        print_header("电源与性能模式")
        
        # NVPModel 模式
        ok, out, _ = run_cmd("nvpmodel -q 2>/dev/null | grep 'Power Mode'")
        if ok and out:
            print_result("电源模式", True, out)
        else:
            ok, out, _ = run_cmd("cat /etc/nvpmodel.conf 2>/dev/null | grep MAXN")
            print_result("电源模式", ok, "MAXN" if ok else "未知")
        
        # Jetson Clocks
        ok, out, _ = run_cmd("systemctl is-active jetson_clocks 2>/dev/null")
        if 'active' in out:
            print_result("Jetson Clocks", True, "已启用（最大性能）")
        else:
            print_result("Jetson Clocks", False, "未启用")
            print("    提示: 运行 'sudo jetson_clocks' 启用最大性能")
        
        # CPU 频率
        ok, out, _ = run_cmd("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null")
        if ok and out:
            freq_mhz = int(out) // 1000
            print_result("CPU 频率", True, f"{freq_mhz} MHz")
        
        # GPU 频率
        ok, out, _ = run_cmd("cat /sys/devices/gpu.0/devfreq/*/cur_freq 2>/dev/null | head -1")
        if ok and out:
            freq_mhz = int(out) // 1000000
            print_result("GPU 频率", True, f"{freq_mhz} MHz")
        
        # 温度
        ok, out, _ = run_cmd("cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | head -1")
        if ok and out:
            temp_c = int(out) / 1000
            status = "正常" if temp_c < 70 else "偏高" if temp_c < 85 else "过热!"
            print_result("CPU 温度", temp_c < 85, f"{temp_c:.1f}°C ({status})")
        
        return True
    
    def generate_report(self):
        """生成测试报告"""
        print_header("测试报告摘要")
        
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"测试耗时: {duration:.1f} 秒")
        print(f"测试时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n关键配置:")
        print(f"  设备: {self.results.get('model', 'N/A')}")
        print(f"  JetPack: {self.results.get('jetpack', 'N/A')}")
        print(f"  CUDA: {self.results.get('cuda', 'N/A')}")
        print(f"  cuDNN: {self.results.get('cudnn', 'N/A')}")
        print(f"  TensorRT: {self.results.get('tensorrt', 'N/A')}")
        print(f"  Python: {self.results.get('python', 'N/A')}")
        print(f"  内存: {self.results.get('memory', 'N/A')}")
        print(f"  磁盘可用: {self.results.get('disk_free', 'N/A')}")
        
        # 检查关键依赖
        print("\n依赖状态:")
        critical_pkgs = ['torch', 'opencv-python', 'tensorrt']
        all_ok = True
        for pkg in critical_pkgs:
            version = self.results.get(f'pkg_{pkg}')
            if version:
                print(f"  ✓ {pkg}: {version}")
            else:
                print(f"  ✗ {pkg}: 未安装")
                all_ok = False
        
        # 总体评估
        print("\n" + "=" * 60)
        if all_ok:
            print("✓ Jetson Nano 环境就绪，可以进行模型部署！")
        else:
            print("⚠ 部分依赖缺失，请安装后再进行部署")
            print("\n建议安装命令:")
            if not self.results.get('pkg_torch'):
                print("  # PyTorch for Jetson")
                print("  wget https://nvidia.box.com/shared/static/xxx.whl")
                print("  pip3 install torch-xxx.whl")
            if not self.results.get('pkg_opencv-python'):
                print("  sudo apt install python3-opencv")
        print("=" * 60)
        
        return self.results
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("  Jetson Nano 功能完整度测试")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 60)
        
        self.test_system_info()
        self.test_memory_storage()
        self.test_cuda_environment()
        self.test_python_environment()
        self.test_camera()
        self.test_network()
        self.test_power_mode()
        self.test_inference_performance()
        self.generate_report()
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Jetson Nano 功能完整度测试")
    parser.add_argument('command', nargs='?', default='all',
                       choices=['all', 'system', 'memory', 'cuda', 'python', 
                               'camera', 'network', 'power', 'inference', 'report'],
                       help='测试命令')
    
    args = parser.parse_args()
    tester = JetsonTester()
    
    if args.command == 'all':
        tester.run_all_tests()
    elif args.command == 'system':
        tester.test_system_info()
    elif args.command == 'memory':
        tester.test_memory_storage()
    elif args.command == 'cuda':
        tester.test_cuda_environment()
    elif args.command == 'python':
        tester.test_python_environment()
    elif args.command == 'camera':
        tester.test_camera()
    elif args.command == 'network':
        tester.test_network()
    elif args.command == 'power':
        tester.test_power_mode()
    elif args.command == 'inference':
        tester.test_inference_performance()
    elif args.command == 'report':
        tester.generate_report()


if __name__ == "__main__":
    main()
