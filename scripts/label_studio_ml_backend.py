#!/usr/bin/env python3
"""
Label Studio ML Backend
集成 YOLO11 模型提供半自动标注
"""

import os
import io
import logging
from typing import List, Dict, Optional
from PIL import Image
import numpy as np

try:
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.utils import get_single_tag_keys
    from ultralytics import YOLO
except ImportError as e:
    print(f"错误: {e}")
    print("请运行: pip install label-studio-ml ultralytics")
    exit(1)

logger = logging.getLogger(__name__)


class YOLOBackend(LabelStudioMLBase):
    """YOLO 模型后端，用于 Label Studio 半自动标注"""
    
    def __init__(self, model_path: str = "yolo11n.pt", **kwargs):
        """
        初始化 YOLO 后端
        
        Args:
            model_path: YOLO 模型路径
        """
        super(YOLOBackend, self).__init__(**kwargs)
        
        self.model_path = model_path
        self.conf_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.25'))
        self.iou_threshold = float(os.getenv('IOU_THRESHOLD', '0.45'))
        
        # 加载模型
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        
        # 类别映射
        self.label_map = {
            0: 'wire',
            1: 'slipper', 
            2: 'sock',
            3: 'cable',
            4: 'toy',
            5: 'obstacle'
        }
        
        # 从模型获取类别（如果可用）
        if hasattr(self.model, 'names'):
            self.label_map = self.model.names
        
        logger.info(f"Model loaded. Classes: {self.label_map}")
        logger.info(f"Confidence threshold: {self.conf_threshold}")
        logger.info(f"IOU threshold: {self.iou_threshold}")
    
    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        对任务进行预测
        
        Args:
            tasks: Label Studio 任务列表
            
        Returns:
            预测结果列表
        """
        predictions = []
        
        for task in tasks:
            # 获取图像 URL
            image_url = task['data'].get('image')
            if not image_url:
                logger.warning(f"Task {task.get('id')} has no image")
                continue
            
            try:
                # 加载图像
                image = self._load_image(image_url)
                if image is None:
                    continue
                
                # 运行推理
                results = self.model.predict(
                    source=image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # 转换为 Label Studio 格式
                result = results[0]
                boxes = result.boxes
                
                if len(boxes) == 0:
                    logger.info(f"No detections for task {task.get('id')}")
                    predictions.append({
                        'result': [],
                        'score': 0.0,
                        'model_version': self.model_path
                    })
                    continue
                
                # 获取图像尺寸
                img_height, img_width = image.shape[:2]
                
                # 构建预测结果
                result_annotations = []
                total_score = 0.0
                
                for box in boxes:
                    # 获取边界框坐标（归一化）
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 转换为百分比
                    x_percent = (x1 / img_width) * 100
                    y_percent = (y1 / img_height) * 100
                    width_percent = ((x2 - x1) / img_width) * 100
                    height_percent = ((y2 - y1) / img_height) * 100
                    
                    # 获取类别和置信度
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    total_score += conf
                    
                    # 获取类别名称
                    label = self.label_map.get(cls, f"class_{cls}")
                    
                    # Label Studio 标注格式
                    annotation = {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": x_percent,
                            "y": y_percent,
                            "width": width_percent,
                            "height": height_percent,
                            "rotation": 0,
                            "rectanglelabels": [label]
                        },
                        "score": conf
                    }
                    
                    result_annotations.append(annotation)
                
                # 计算平均置信度
                avg_score = total_score / len(boxes) if len(boxes) > 0 else 0.0
                
                predictions.append({
                    'result': result_annotations,
                    'score': avg_score,
                    'model_version': self.model_path
                })
                
                logger.info(f"Task {task.get('id')}: {len(boxes)} detections, avg score: {avg_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing task {task.get('id')}: {e}")
                predictions.append({
                    'result': [],
                    'score': 0.0,
                    'model_version': self.model_path
                })
        
        return predictions
    
    def _load_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            image_url: 图像 URL 或本地路径
            
        Returns:
            图像数组或 None
        """
        try:
            # 处理本地文件路径
            if image_url.startswith('file://'):
                image_path = image_url[7:]
            elif image_url.startswith('/'):
                image_path = image_url
            else:
                # TODO: 处理 HTTP URL
                logger.warning(f"HTTP URLs not yet supported: {image_url}")
                return None
            
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Error loading image {image_url}: {e}")
            return None
    
    def fit(self, tasks: List[Dict], **kwargs) -> Dict:
        """
        训练模型（可选）
        
        Args:
            tasks: 标注任务列表
            
        Returns:
            训练结果
        """
        logger.info(f"Fit called with {len(tasks)} tasks")
        # 这里可以实现在线学习或微调
        return {
            'status': 'ok',
            'message': 'Model fit not implemented yet'
        }


def init_app(**kwargs):
    """初始化应用"""
    model_path = os.getenv('MODEL_PATH', 'yolo11n.pt')
    return YOLOBackend(model_path=model_path, **kwargs)


if __name__ == '__main__':
    # 测试代码
    backend = YOLOBackend(model_path='yolo11n.pt')
    
    test_task = {
        'id': 1,
        'data': {
            'image': 'data/seed_dataset_v2/seed_0001.jpg'
        }
    }
    
    predictions = backend.predict([test_task])
    print(f"Predictions: {predictions}")
