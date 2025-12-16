# 标注工具集成指南

## 🎯 目标
集成 LabelImg 和 Label Studio 实现高效的半自动标注流程

## 📦 已创建的文件

### 1. 脚本文件
- `scripts/setup_annotation_tools.sh` - 工具安装脚本
- `scripts/auto_annotate.sh` - 半自动标注 Shell 包装器
- `scripts/auto_annotate.py` - 半自动标注 Python 实现
- `scripts/label_studio_ml_backend.py` - Label Studio ML Backend
- `scripts/start_label_studio.sh` - Label Studio 启动脚本

### 2. 配置文件
- `label_studio/config.xml` - Label Studio 标注界面配置
- `label_studio/README.md` - 完整使用文档

## 🚀 快速开始

### 方式 1: LabelImg (快速本地标注)

```bash
# 1. 安装 LabelImg
conda activate obstacle_detection
pip install labelImg

# 2. 启动标注
labelImg data/seed_dataset_v2 configs/classes.txt

# 快捷键:
# W - 创建框, D - 下一张, A - 上一张, Ctrl+S - 保存
```

### 方式 2: 半自动标注流程 (推荐)

```bash
# 1. 安装工具
bash scripts/setup_annotation_tools.sh

# 2. 运行预标注 (使用 YOLO11n 生成初步标注)
python scripts/auto_annotate.py \
    --input data/seed_dataset_v2 \
    --output data/seed_dataset_v2/auto_labels \
    --model yolo11n.pt \
    --conf 0.25 \
    --visualize

# 3. 使用 LabelImg 检查和修正
labelImg data/seed_dataset_v2 data/seed_dataset_v2/auto_labels

# 或使用快捷脚本
bash scripts/auto_annotate.sh data/seed_dataset_v2
```

### 方式 3: Label Studio (团队协作)

```bash
# 1. 安装 Label Studio
pip install label-studio label-studio-ml

# 2. 启动服务
bash scripts/start_label_studio.sh

# 3. 浏览器打开 http://localhost:8080
# 4. 创建项目并导入数据
```

## 💡 推荐工作流

### 个人开发者
```
预标注 → LabelImg 修正 → 训练模型
  ↓
YOLO11n  →  快速检查  →  开始训练
(2分钟)      (30分钟)      (2小时)
```

### 团队协作
```
预标注 → Label Studio 分配 → 多人标注 → 审核 → 导出
  ↓          ↓                ↓         ↓       ↓
YOLO11n   任务分配         并行标注   质量控制  YOLO格式
```

## 📊 效率提升

| 方法 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| 纯手工 | 100张/4小时 | ⭐⭐⭐⭐⭐ | 少量数据 |
| LabelImg | 100张/3小时 | ⭐⭐⭐⭐ | 个人开发 |
| 预标注+LabelImg | 100张/1小时 | ⭐⭐⭐⭐ | **推荐** |
| Label Studio | 100张/2小时 | ⭐⭐⭐⭐⭐ | 团队协作 |
| 预标注+Label Studio | 100张/30分钟 | ⭐⭐⭐⭐⭐ | **最佳** |

## 🔧 核心功能

### auto_annotate.py 功能
- ✅ 加载 YOLO 预训练模型
- ✅ 批量推理生成预标注
- ✅ 保存 YOLO 格式 (.txt)
- ✅ 可选可视化输出
- ✅ 置信度筛选
- ✅ 类别统计

### Label Studio ML Backend 功能
- ✅ 实时预测接口
- ✅ 与 Label Studio 集成
- ✅ 动态置信度调整
- ✅ 支持在线学习（预留接口）

## 📋 标注类别

当前配置的障碍物类别:
1. **wire** (电线) - 红色
2. **slipper** (拖鞋) - 蓝色
3. **sock** (袜子) - 绿色
4. **cable** (数据线) - 黄色
5. **toy** (小玩具) - 紫色
6. **obstacle** (其他障碍物) - 橙色

修改类别: 编辑 `configs/classes.txt` 和 `label_studio/config.xml`

## 🎓 使用示例

### 示例 1: 快速测试预标注
```bash
# 对10张图像进行预标注测试
python scripts/auto_annotate.py \
    --input data/seed_dataset_v2 \
    --output test_labels \
    --model yolo11n.pt \
    --conf 0.3 \
    --visualize

# 查看可视化结果
open test_labels/visualizations/
```

### 示例 2: 批量预标注
```bash
# 对全部200张进行预标注
bash scripts/auto_annotate.sh data/seed_dataset_v2

# 启动 LabelImg 检查
labelImg data/seed_dataset_v2 data/seed_dataset_v2/auto_labels
```

### 示例 3: Label Studio 完整流程
```bash
# 1. 启动服务 (带 ML Backend)
bash scripts/start_label_studio.sh

# 2. 在浏览器中:
#    - 创建项目: "Obstacle Detection"
#    - 导入配置: label_studio/config.xml
#    - 添加数据: data/seed_dataset_v2
#    - 连接 ML Backend: http://localhost:9090

# 3. 开始标注 (按空格键快速提交)

# 4. 导出结果
#    Export -> YOLO -> Download
```

## 🐛 故障排除

### LabelImg 启动失败
```bash
# macOS
brew install pyqt5
pip install labelImg

# Linux
sudo apt-get install pyqt5-dev-tools
pip install labelImg
```

### Label Studio 端口占用
```bash
# 检查占用
lsof -i :8080

# 使用其他端口
label-studio start --port 8081
```

### 预标注结果为空
```bash
# 降低置信度阈值
python scripts/auto_annotate.py --input data/seed_dataset_v2 --conf 0.1

# 检查模型加载
python -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); print(m.names)"
```

## 📚 参考文档

- LabelImg: https://github.com/heartexlabs/labelImg
- Label Studio: https://labelstud.io/
- YOLO 格式: https://docs.ultralytics.com/datasets/detect/
- 完整 Label Studio 指南: `label_studio/README.md`

## ✅ 下一步行动

当前数据集状态:
- ✅ 992 帧已提取
- ✅ 200 张种子数据集已创建
- ✅ 标注工具已配置
- ⏳ **待标注: 200 张图像**

推荐开始标注:
```bash
# 推荐流程 (1小时完成200张)
bash scripts/auto_annotate.sh data/seed_dataset_v2
labelImg data/seed_dataset_v2 data/seed_dataset_v2/auto_labels
```

标注完成后:
```bash
# 开始训练
bash scripts/train_day2.sh train yolo11n.pt 50
```
