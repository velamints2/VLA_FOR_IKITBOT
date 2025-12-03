"""
测试RGBD摄像头视频流获取
用于验证RealSense或其他RGBD相机的连接和数据读取
"""
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("错误: OpenCV未安装。请运行: pip install opencv-python")
    sys.exit(1)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    print("警告: pyrealsense2未安装。如果使用RealSense相机，请运行: pip install pyrealsense2")
    REALSENSE_AVAILABLE = False


def test_realsense_camera():
    """测试RealSense摄像头"""
    if not REALSENSE_AVAILABLE:
        print("跳过RealSense测试（库未安装）")
        return False
    
    print("=" * 50)
    print("测试RealSense RGBD摄像头...")
    print("=" * 50)
    
    try:
        # 配置RealSense管道
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 启用RGB和深度流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 启动管道
        print("正在连接摄像头...")
        pipeline.start(config)
        print("✓ 摄像头连接成功！")
        
        # 获取几帧测试
        print("正在获取测试帧...")
        for i in range(5):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print(f"✗ 第{i+1}帧获取失败")
                continue
            
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            print(f"✓ 第{i+1}帧: RGB {color_image.shape}, Depth {depth_image.shape}")
        
        # 停止管道
        pipeline.stop()
        print("✓ RealSense测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ RealSense测试失败: {e}")
        return False


def test_webcam():
    """测试普通网络摄像头（备用方案）"""
    print("\n" + "=" * 50)
    print("测试普通摄像头...")
    print("=" * 50)
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ 无法打开摄像头")
            return False
        
        print("✓ 摄像头打开成功")
        
        # 读取几帧测试
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"✓ 第{i+1}帧: {frame.shape}")
            else:
                print(f"✗ 第{i+1}帧获取失败")
        
        cap.release()
        print("✓ 普通摄像头测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 摄像头测试失败: {e}")
        return False


def test_video_file(video_path):
    """测试视频文件读取"""
    print("\n" + "=" * 50)
    print(f"测试视频文件: {video_path}")
    print("=" * 50)
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("✗ 无法打开视频文件")
            return False
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ 视频信息:")
        print(f"  - 分辨率: {width}x{height}")
        print(f"  - 帧率: {fps} FPS")
        print(f"  - 总帧数: {frame_count}")
        
        # 读取前几帧
        for i in range(min(5, frame_count)):
            ret, frame = cap.read()
            if ret:
                print(f"✓ 第{i+1}帧: {frame.shape}")
            else:
                print(f"✗ 第{i+1}帧读取失败")
        
        cap.release()
        print("✓ 视频文件测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 视频文件测试失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RGBD摄像头测试脚本")
    print("=" * 60)
    
    # 测试RealSense
    realsense_ok = test_realsense_camera()
    
    # 如果RealSense失败，测试普通摄像头
    if not realsense_ok:
        print("\n提示: RealSense不可用，尝试使用普通摄像头...")
        webcam_ok = test_webcam()
    else:
        webcam_ok = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if realsense_ok:
        print("✓ RealSense RGBD摄像头工作正常")
        print("  可以获取RGB和深度数据")
    elif webcam_ok:
        print("⚠ 仅普通摄像头可用（无深度信息）")
        print("  建议：连接RealSense或使用预录制的RGBD视频")
    else:
        print("✗ 无可用摄像头")
        print("  请检查：")
        print("  1. RealSense相机是否正确连接")
        print("  2. 驱动程序是否安装")
        print("  3. 摄像头权限是否允许")
    
    return 0 if (realsense_ok or webcam_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
