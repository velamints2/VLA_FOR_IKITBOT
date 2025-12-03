"""
障碍物检测类别定义
"""

# 类别ID到类别名的映射
CLASS_NAMES = {
    0: 'wire',          # 电线/数据线
    1: 'shoe',          # 拖鞋/鞋子
    2: 'small_object'   # 小物体（玩具、杂物等）
}

# 类别总数
NUM_CLASSES = len(CLASS_NAMES)

# YOLO格式的类别名列表
YOLO_CLASSES = ['wire', 'shoe', 'small_object']
