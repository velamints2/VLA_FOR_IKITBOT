"""
YOLO训练脚本
用于训练障碍物检测模型
"""
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: ultralytics未安装")
    print("请运行: pip install ultralytics")
    exit(1)


def train_model(
    data_config,
    model='yolov10n.pt',
    epochs=100,
    imgsz=640,
    batch=16,
    device='0',
    project='runs/train',
    name='obstacle_detection'
):
    """
    训练YOLO模型
    
    Args:
        data_config: 数据配置文件路径
        model: 模型名称或预训练权重路径
        epochs: 训练轮数
        imgsz: 输入图像大小
        batch: batch大小
        device: 设备（0=GPU, cpu=CPU）
        project: 项目保存路径
        name: 实验名称
    """
    print("=" * 60)
    print("YOLO训练配置")
    print("=" * 60)
    print(f"数据配置: {data_config}")
    print(f"模型: {model}")
    print(f"训练轮数: {epochs}")
    print(f"图像大小: {imgsz}")
    print(f"Batch大小: {batch}")
    print(f"设备: {device}")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    yolo_model = YOLO(model)
    
    # 开始训练
    print("\n开始训练...")
    results = yolo_model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=50,        # 早停耐心值
        save=True,          # 保存检查点
        save_period=10,     # 每10轮保存一次
        val=True,           # 训练时验证
        plots=True,         # 生成训练图表
        verbose=True        # 详细输出
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
    print(f"最终模型: {results.save_dir}/weights/last.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="训练YOLO障碍物检测模型")
    parser.add_argument('--data', type=str, default='configs/data.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--model', type=str, default='yolov10n.pt',
                       help='模型名称（yolov10n.pt, yolo11n.pt等）')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像大小')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch大小')
    parser.add_argument('--device', type=str, default='0',
                       help='设备（0, 1, 2... 或 cpu）')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='项目保存路径')
    parser.add_argument('--name', type=str, default='obstacle_detection',
                       help='实验名称')
    
    args = parser.parse_args()
    
    # 训练
    train_model(
        data_config=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
