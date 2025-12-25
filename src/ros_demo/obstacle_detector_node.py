#!/usr/bin/env python3
"""
ROS1 障碍物检测演示节点
实时处理摄像头视频流并显示检测结果

功能:
1. 订阅摄像头图像话题
2. 使用YOLO模型进行障碍物检测
3. 发布带标注的检测结果图像
4. 发布检测信息（类别、置信度）

使用方法:
    rosrun obstacle_detection obstacle_detector_node.py
    
话题:
    订阅: /camera/image_raw (sensor_msgs/Image)
    发布: /obstacle_detection/result (sensor_msgs/Image)
    发布: /obstacle_detection/info (std_msgs/String)
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge, CvBridgeError
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("警告: ROS 相关包未安装，仅支持独立模式运行")

import cv2
import numpy as np
from ultralytics import YOLO


class ObstacleDetectorNode:
    """ROS 障碍物检测节点"""
    
    def __init__(self, model_path: str = None, standalone: bool = False):
        """
        初始化检测节点
        
        Args:
            model_path: YOLO模型路径，默认使用 models/best.pt
            standalone: 是否为独立模式（不使用ROS）
        """
        self.standalone = standalone or not ROS_AVAILABLE
        
        # 模型路径
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "best.pt")
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 推理参数
        self.conf_threshold = 0.5  # 置信度阈值
        self.imgsz = 640           # 输入图像大小
        self.device = 'cpu'        # 推理设备 (cpu/cuda/mps)
        
        # 检测统计
        self.frame_count = 0
        self.detection_count = 0
        
        if not self.standalone:
            self._init_ros()
        else:
            print("运行在独立模式（无ROS）")
    
    def _init_ros(self):
        """初始化ROS节点和话题"""
        rospy.init_node('obstacle_detector_demo', anonymous=True)
        
        # 图像转换桥接
        self.bridge = CvBridge()
        
        # 从参数服务器获取配置
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.5)
        self.imgsz = rospy.get_param('~imgsz', 640)
        self.device = rospy.get_param('~device', 'cpu')
        input_topic = rospy.get_param('~input_topic', '/camera/image_raw')
        output_topic = rospy.get_param('~output_topic', '/obstacle_detection/result')
        info_topic = rospy.get_param('~info_topic', '/obstacle_detection/info')
        
        # 订阅摄像头话题
        self.image_sub = rospy.Subscriber(
            input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24  # 增大缓冲区以处理大图像
        )
        
        # 发布检测结果图像
        self.result_pub = rospy.Publisher(
            output_topic,
            Image,
            queue_size=1
        )
        
        # 发布检测信息
        self.info_pub = rospy.Publisher(
            info_topic,
            String,
            queue_size=10
        )
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("障碍物检测演示节点已启动")
        rospy.loginfo(f"  模型: {self.model.model_name}")
        rospy.loginfo(f"  输入话题: {input_topic}")
        rospy.loginfo(f"  输出话题: {output_topic}")
        rospy.loginfo(f"  置信度阈值: {self.conf_threshold}")
        rospy.loginfo(f"  推理设备: {self.device}")
        rospy.loginfo("=" * 50)
    
    def detect(self, image: np.ndarray) -> tuple:
        """
        对图像进行障碍物检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            annotated_frame: 带标注的图像
            detections: 检测结果列表
        """
        # 使用YOLO模型进行推理
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        
        # 获取检测结果
        result = results[0]
        annotated_frame = result.plot()
        
        # 解析检测结果
        detections = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': xyxy.tolist()  # [x1, y1, x2, y2]
                })
        
        return annotated_frame, detections
    
    def image_callback(self, msg: Image):
        """
        ROS图像回调函数
        
        Args:
            msg: ROS Image消息
        """
        try:
            # 转换ROS图像为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 进行检测
            annotated_frame, detections = self.detect(cv_image)
            
            # 更新统计
            self.frame_count += 1
            self.detection_count += len(detections)
            
            # 发布带标注的图像
            result_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            result_msg.header = msg.header
            self.result_pub.publish(result_msg)
            
            # 发布检测信息
            if detections:
                detection_strs = [f"{d['class_name']}: {d['confidence']:.2f}" for d in detections]
                info_msg = f"帧 {self.frame_count}: 检测到 {len(detections)} 个障碍物 [{', '.join(detection_strs)}]"
            else:
                info_msg = f"帧 {self.frame_count}: 未检测到障碍物"
            
            self.info_pub.publish(info_msg)
            
            # 每100帧打印一次统计
            if self.frame_count % 100 == 0:
                rospy.loginfo(f"已处理 {self.frame_count} 帧，累计检测 {self.detection_count} 个障碍物")
                
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge 错误: {e}")
        except Exception as e:
            rospy.logerr(f"图像处理错误: {e}")
    
    def run_standalone(self, source: str = "0"):
        """
        独立模式运行（不依赖ROS）
        
        Args:
            source: 视频源 (摄像头ID或视频文件路径)
        """
        print(f"独立模式运行，视频源: {source}")
        
        # 打开视频源
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频源 {source}")
            return
        
        print("按 'q' 键退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频结束或读取失败")
                break
            
            # 进行检测
            annotated_frame, detections = self.detect(frame)
            
            # 显示检测信息
            info_text = f"检测到 {len(detections)} 个障碍物"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Obstacle Detection', annotated_frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                save_path = f"detection_{self.frame_count}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"已保存: {save_path}")
            
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"共处理 {self.frame_count} 帧")
    
    def run(self):
        """运行ROS节点"""
        if self.standalone:
            self.run_standalone()
        else:
            rospy.spin()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ROS障碍物检测演示节点")
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='YOLO模型路径 (默认: models/best.pt)')
    parser.add_argument('--standalone', '-s', action='store_true',
                       help='独立模式运行（不使用ROS）')
    parser.add_argument('--source', type=str, default='0',
                       help='独立模式视频源 (摄像头ID或视频路径)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='cpu',
                       help='推理设备 (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    try:
        node = ObstacleDetectorNode(
            model_path=args.model,
            standalone=args.standalone
        )
        node.conf_threshold = args.conf
        node.device = args.device
        
        if args.standalone:
            node.run_standalone(args.source)
        else:
            node.run()
            
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        raise


if __name__ == '__main__':
    main()

