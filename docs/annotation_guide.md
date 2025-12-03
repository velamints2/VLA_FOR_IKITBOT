# 标注工作流程指南

## 前提条件
- 已安装 LabelImg 标注工具
- 已提取并精选种子数据集图像

## 步骤1：安装 LabelImg

### 方法1：使用pip（推荐）
```bash
conda activate obstacle_detection
pip install labelImg
```

### 方法2：从源码安装
```bash
git clone https://github.com/heartexlabs/labelImg
cd labelImg
pip install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 labelImg.py
```

## 步骤2：启动 LabelImg

```bash
cd /Users/macbookair/Documents/trae_projects/llm

# 启动 LabelImg
labelImg data/seed_dataset/images data/seed_dataset/labels
```

## 步骤3：配置 LabelImg

1. **更改保存目录**
   - File → Change Save Dir → `data/seed_dataset/labels`

2. **选择YOLO格式**
   - View → 勾选 "Use YOLO format"
   - 或快捷键: Ctrl+Y

3. **加载类别定义**
   - 在 `data/seed_dataset` 创建 `classes.txt`
   - 内容：
     ```
     wire
     shoe
     small_object
     ```

## 步骤4：标注流程

### 快捷键
- `W`: 创建矩形框
- `A`: 上一张图片
- `D`: 下一张图片
- `Ctrl+S`: 保存
- `Del`: 删除选中的框

### 标注原则

#### 1. wire（电线/数据线）
- 标注整个电线的可见部分
- 包括弯曲、缠绕的部分
- 尽量框选完整

#### 2. shoe（拖鞋/鞋子）
- 标注整只鞋子
- 包括鞋帮、鞋底
- 即使部分遮挡也要标注

#### 3. small_object（小物体）
- 玩具、杂物、小型障碍物
- 标注物体主体部分
- 太小（<20像素）的物体可以跳过

### 标注技巧

1. **优先级**
   - 先标注明显、清晰的障碍物
   - 边界模糊的可以暂时跳过
   - 目标：快速建立50-100个样本

2. **质量控制**
   - 边框应紧贴物体边缘
   - 避免过大或过小的框
   - 重叠物体分别标注

3. **效率提升**
   - 使用快捷键加速
   - 连续标注相似场景
   - 每标注10张保存一次

## 步骤5：验证标注

标注完成后，运行验证脚本：
```bash
python src/data_processing/validate_annotations.py \
    --images data/seed_dataset/images \
    --labels data/seed_dataset/labels
```

## 步骤6：可视化检查

查看标注效果：
```bash
python src/data_processing/visualize_annotations.py \
    --images data/seed_dataset/images \
    --labels data/seed_dataset/labels \
    --output docs/annotation_samples.png
```

## 预期输出

标注完成后的目录结构：
```
data/seed_dataset/
├── images/
│   ├── seed_0000.jpg
│   ├── seed_0001.jpg
│   └── ...
├── labels/
│   ├── seed_0000.txt
│   ├── seed_0001.txt
│   └── ...
├── classes.txt
├── train.txt
└── val.txt
```

## YOLO标注格式

每个 `.txt` 文件对应一张图片，格式：
```
<class_id> <x_center> <y_center> <width> <height>
```

所有坐标都是归一化的（0-1之间）。

示例：
```
0 0.5 0.6 0.3 0.1    # wire
1 0.3 0.8 0.2 0.15   # shoe
2 0.7 0.7 0.1 0.1    # small_object
```

## 常见问题

### Q1: LabelImg无法启动
```bash
# 检查安装
pip show labelImg

# 重新安装
pip uninstall labelImg
pip install labelImg
```

### Q2: 保存格式不对
- 确保选择了 "Use YOLO format"
- 检查 classes.txt 是否在正确位置

### Q3: 标注框不准确
- 放大图像再标注（鼠标滚轮）
- 使用键盘微调框位置

## 下一步

标注完成后：
1. 运行数据集划分：`python src/data_processing/split_dataset.py`
2. 开始第2天训练任务
