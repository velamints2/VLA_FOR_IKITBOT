"""
环境检查脚本 - 验证训练和部署环境
"""
import sys
import subprocess


def check_cuda_availability():
    """检查CUDA版本和GPU可用性"""
    print("=" * 50)
    print("检查CUDA环境...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠ 警告: CUDA不可用，将使用CPU训练（速度较慢）")
        
        return torch.cuda.is_available()
    
    except ImportError:
        print("✗ 错误: PyTorch未安装")
        return False


def verify_training_env():
    """验证训练环境的所有依赖"""
    print("\n" + "=" * 50)
    print("检查训练环境依赖...")
    print("=" * 50)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'ultralytics': 'Ultralytics YOLO',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: 未安装")
            all_installed = False
    
    return all_installed


def run_simple_test():
    """运行简单的PyTorch测试"""
    print("\n" + "=" * 50)
    print("运行简单测试...")
    print("=" * 50)
    
    try:
        import torch
        import torchvision
        
        # 创建一个简单的tensor并测试GPU
        x = torch.rand(3, 224, 224)
        print(f"✓ 创建测试tensor: {x.shape}")
        
        if torch.cuda.is_available():
            x = x.cuda()
            print(f"✓ Tensor移至GPU: {x.device}")
        
        # 测试一个简单的卷积操作
        conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        if torch.cuda.is_available():
            conv = conv.cuda()
        
        with torch.no_grad():
            y = conv(x.unsqueeze(0))
        
        print(f"✓ 卷积测试成功: 输入{x.shape} -> 输出{y.shape}")
        print("✓ 环境测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("扫地机器人障碍物检测 - 环境验证脚本")
    print("=" * 60)
    
    # 检查Python版本
    print(f"\nPython版本: {sys.version}")
    
    # 依次检查各项
    cuda_ok = check_cuda_availability()
    env_ok = verify_training_env()
    test_ok = run_simple_test()
    
    # 总结
    print("\n" + "=" * 60)
    print("环境检查总结")
    print("=" * 60)
    print(f"CUDA环境: {'✓ 通过' if cuda_ok else '⚠ 未通过（将使用CPU）'}")
    print(f"依赖包: {'✓ 通过' if env_ok else '✗ 未通过'}")
    print(f"功能测试: {'✓ 通过' if test_ok else '✗ 未通过'}")
    
    if env_ok and test_ok:
        print("\n✓ 训练环境配置完成！可以开始训练。")
        return 0
    else:
        print("\n✗ 环境配置存在问题，请根据上述提示修复。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
