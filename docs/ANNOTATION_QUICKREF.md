# 🏷️ 标注工具快速参考

## ⚡ 快速开始 (3 分钟)

```bash
# 1. 安装工具
bash scripts/setup_annotation_tools.sh

# 2. 运行演示
bash scripts/demo_annotation.sh
```

## 📋 三种标注方式对比

| 方式 | 命令 | 时间 | 适用场景 |
|------|------|------|----------|
| **纯手工** | `labelImg data/seed_dataset_v2` | 3-4h | 初学者 |
| **半自动** ⭐ | `bash scripts/auto_annotate.sh` + `labelImg` | 1h | **推荐** |
| **Label Studio** | `bash scripts/start_label_studio.sh` | 1-2h | 团队协作 |

## 🎯 推荐工作流 (200张/1小时)

```bash
# Step 1: 预标注 (2分钟)
bash scripts/auto_annotate.sh data/seed_dataset_v2

# Step 2: 检查修正 (50分钟)
labelImg data/seed_dataset_v2 data/seed_dataset_v2/auto_labels

# Step 3: 开始训练 (2小时)
bash scripts/train_day2.sh train yolo11n.pt 50
```

## 🔧 常用命令

### LabelImg
```bash
# 基础使用
labelImg [图像目录] [标注目录]

# 完整示例
labelImg data/seed_dataset_v2 data/seed_dataset_v2/auto_labels

# 指定类别文件
labelImg data/seed_dataset_v2 configs/classes.txt
```

**快捷键:**
- `W` - 创建矩形框
- `D` - 下一张
- `A` - 上一张
- `Ctrl+S` - 保存
- `Del` - 删除框

### 半自动标注
```bash
# 使用默认参数
python scripts/auto_annotate.py -i data/seed_dataset_v2 -o data/auto_labels

# 调整置信度阈值
python scripts/auto_annotate.py -i data/seed_dataset_v2 -o data/auto_labels --conf 0.3

# 启用可视化
python scripts/auto_annotate.py -i data/seed_dataset_v2 -o data/auto_labels --visualize
```

### Label Studio
```bash
# 启动服务
label-studio start

# 带 ML Backend
bash scripts/start_label_studio.sh

# 指定端口
label-studio start --port 8081
```

## 🏷️ 标注类别

| ID | 类别 | 颜色 | 说明 | 快捷键 |
|----|------|------|------|--------|
| 0 | wire | 🔴 红 | 电线 | 1 |
| 1 | slipper | 🔵 蓝 | 拖鞋 | 2 |
| 2 | sock | 🟢 绿 | 袜子 | 3 |
| 3 | cable | 🟡 黄 | 数据线 | 4 |
| 4 | toy | 🟣 紫 | 小玩具 | 5 |
| 5 | obstacle | 🟠 橙 | 其他障碍物 | 6 |

## 📊 效率提升技巧

### 1. 批量处理
```bash
# 按目录批量预标注
for dir in data/frames/*/; do
    python scripts/auto_annotate.py -i "$dir" -o "${dir}/labels"
done
```

### 2. 置信度优化
```bash
# 高置信度 (减少误检)
python scripts/auto_annotate.py --conf 0.4

# 低置信度 (增加召回)
python scripts/auto_annotate.py --conf 0.15
```

### 3. 并行标注
- 使用 Label Studio 分配任务给多人
- 设置审核流程确保质量
- 导出统一格式

## 🐛 常见问题

### LabelImg 无法启动
```bash
# macOS
brew install pyqt5
pip install labelImg

# Linux
sudo apt-get install pyqt5-dev-tools
pip install labelImg
```

### 预标注结果为空
```bash
# 降低置信度
python scripts/auto_annotate.py --conf 0.1

# 检查模型
python -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); print(m.names)"
```

### Label Studio 端口占用
```bash
# 查看占用
lsof -i :8080

# 换端口
label-studio start --port 8081
```

## 📚 更多文档

- **完整指南**: `docs/annotation_tools_guide.md`
- **Label Studio 教程**: `label_studio/README.md`
- **项目文档**: `README.md`

## 💡 最佳实践

### ✅ 标注质量检查清单
- [ ] 所有目标都已标注
- [ ] 边界框紧贴目标
- [ ] 类别标注正确
- [ ] 无重复标注
- [ ] 遮挡目标已标注可见部分
- [ ] 小目标（< 32x32）已标注
- [ ] 边缘目标已完整标注

### ⚡ 效率最大化
1. **预标注优先**: 先跑 auto_annotate，再人工修正
2. **快捷键熟练**: W创建、D下一张、Ctrl+S保存
3. **分批标注**: 每50张一批，避免疲劳
4. **质量抽查**: 每标完50张抽查10张
5. **备份标注**: 定期 git commit 保存进度

### 🎯 标注规范
- **边界框**: 紧贴目标，包含完整轮廓
- **遮挡处理**: 标注可见部分
- **小目标**: 尽量标注，即使模糊
- **边缘目标**: 如在图像边缘被裁切，标注可见部分
- **类别选择**: 不确定时选 `obstacle`

## 🚀 下一步

标注完成后:
```bash
# 验证标注
python src/data_processing/validate_annotations.py data/seed_dataset_v2

# 划分数据集
python src/data_processing/split_dataset.py data/seed_dataset_v2 --train-ratio 0.8

# 开始训练
bash scripts/train_day2.sh train yolo11n.pt 50
```

---

**提示**: 遇到问题查看完整文档或运行演示 `bash scripts/demo_annotation.sh`
