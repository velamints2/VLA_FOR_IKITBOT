# Label Studio 配置和使用指南

## 🎯 Label Studio 介绍

Label Studio 是一个开源的数据标注平台，支持：
- 🌐 Web 界面，支持多人协作
- 🤖 集成机器学习模型进行半自动标注
- 📊 标注质量控制和审核
- 🔄 多种数据格式导入/导出

## 📦 安装

### 方法 1: 使用自动安装脚本
```bash
bash scripts/setup_annotation_tools.sh
```

### 方法 2: 手动安装
```bash
conda activate obstacle_detection
pip install label-studio label-studio-ml
```

## 🚀 快速开始

### 1. 启动 Label Studio
```bash
# 在项目根目录运行
label-studio start
```

浏览器自动打开 http://localhost:8080

### 2. 创建项目
1. 点击 "Create Project"
2. 项目名称: `Obstacle Detection`
3. 导入配置文件: `label_studio/config.xml`

### 3. 导入数据

#### 方式 A: 本地文件导入
```bash
# 在 Label Studio 界面
Settings -> Cloud Storage -> Add Source Storage
Type: Local Files
Absolute local path: /path/to/data/seed_dataset_v2
```

#### 方式 B: 使用预标注结果
```bash
# 1. 先运行半自动预标注
bash scripts/auto_annotate.sh

# 2. 在 Label Studio 导入时选择:
#    - Images: data/seed_dataset_v2
#    - Pre-annotations: data/seed_dataset_v2/auto_labels
```

## 🤖 半自动标注工作流

### Step 1: 生成预标注
```bash
# 使用 YOLO11n 预训练模型生成初步标注
python scripts/auto_annotate.py \
    --input data/seed_dataset_v2 \
    --output data/seed_dataset_v2/auto_labels \
    --model yolo11n.pt \
    --conf 0.25 \
    --visualize
```

### Step 2: 导入 Label Studio
1. 在 Label Studio 项目中
2. 点击 "Import"
3. 选择 "Import pre-annotated data"
4. 上传图像和对应的标注文件

### Step 3: 人工审核和修正
- ✅ 确认正确的标注
- ✏️ 修正错误的边界框
- ➕ 添加遗漏的目标
- ❌ 删除误检测

### Step 4: 导出标注
```bash
# 在 Label Studio 导出为 YOLO 格式
Export -> YOLO -> Download
```

## 📋 标注类别配置

当前配置的类别（在 `config.xml` 中定义）:

| 类别 | 颜色 | 说明 |
|------|------|------|
| wire | 🔴 红色 | 电线 |
| slipper | 🔵 蓝色 | 拖鞋 |
| sock | 🟢 绿色 | 袜子 |
| cable | 🟡 黄色 | 数据线 |
| toy | 🟣 紫色 | 小玩具 |
| obstacle | 🟠 橙色 | 其他障碍物 |

修改类别: 编辑 `label_studio/config.xml`

## 🔧 高级配置

### 集成自定义模型
```bash
# 创建 ML Backend
label-studio-ml init my_model --script scripts/label_studio_ml_backend.py

# 启动 ML Backend
label-studio-ml start my_model

# 在 Label Studio 中连接
Settings -> Machine Learning -> Add Model
URL: http://localhost:9090
```

### 导出格式
支持多种格式:
- YOLO (txt)
- COCO JSON
- Pascal VOC (xml)
- CSV
- JSON

### 标注统计
```bash
# 在 Label Studio 界面查看
Dashboard -> Statistics
- 标注进度
- 每人标注数量
- 平均标注时间
```

## 💡 最佳实践

### 1. 标注质量控制
- 设置审核流程：至少 2 人审核
- 定期抽查标注质量
- 使用标注指南保持一致性

### 2. 高效标注技巧
- **快捷键使用**:
  - `Space`: 提交当前任务
  - `Ctrl+Enter`: 快速提交
  - `Ctrl+Z`: 撤销
  - `Delete`: 删除选中的框

- **预标注优先**:
  - 先用模型生成初步标注
  - 人工只需修正错误
  - 效率提升 3-5 倍

### 3. 团队协作
- 分配任务给不同标注员
- 设置质量审核人员
- 定期同步标注进度

## 📊 标注进度跟踪

```bash
# 查看标注统计
curl http://localhost:8080/api/projects/1/summary/

# 导出标注数据
curl http://localhost:8080/api/projects/1/export?exportType=YOLO
```

## 🐛 常见问题

### 1. 启动失败
```bash
# 检查端口占用
lsof -i :8080

# 指定其他端口
label-studio start --port 8081
```

### 2. 导入图像失败
- 检查图像路径是否正确
- 确保图像格式支持 (jpg, png)
- 检查文件权限

### 3. 预标注无法导入
- 确保标注文件格式正确
- 检查类别 ID 匹配
- 验证边界框坐标范围 [0, 1]

## 📚 参考资源

- [Label Studio 官方文档](https://labelstud.io/guide/)
- [Label Studio ML Backend](https://github.com/heartexlabs/label-studio-ml-backend)
- [YOLO 格式说明](https://docs.ultralytics.com/datasets/detect/)

## 🔄 工作流示例

```bash
# 完整的半自动标注流程

# 1. 安装工具
bash scripts/setup_annotation_tools.sh

# 2. 生成预标注
bash scripts/auto_annotate.sh data/seed_dataset_v2

# 3. 启动 Label Studio
label-studio start

# 4. 在浏览器中:
#    - 创建项目
#    - 导入图像和预标注
#    - 审核和修正标注

# 5. 导出标注结果
#    Export -> YOLO -> Download

# 6. 开始训练
bash scripts/train_day2.sh train yolo11n.pt 50
```

## ✅ 标注检查清单

- [ ] 所有目标都已标注
- [ ] 边界框紧贴目标
- [ ] 类别标注正确
- [ ] 无重复标注
- [ ] 遮挡目标已标注可见部分
- [ ] 小目标（< 32x32）已标注
- [ ] 边缘目标已完整标注
