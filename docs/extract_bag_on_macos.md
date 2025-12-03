# macOS 上处理 RealSense .bag 文件

## 方法1：使用 Homebrew 安装 librealsense（推荐）

### 步骤1：安装依赖
```bash
# 安装 librealsense
brew install librealsense

# 安装 OpenCV
brew install opencv
```

### 步骤2：编译并运行提取工具
```bash
cd /Users/macbookair/Documents/trae_projects/llm
bash scripts/compile_and_extract.sh
```

这将自动：
1. 编译 C++ 提取工具
2. 处理所有 .bag 文件
3. 提取 RGB 和 Depth 帧到 `data/frames/`

---

## 方法2：手动编译（如果 brew 失败）

### 步骤1：安装依赖
```bash
brew install cmake pkg-config
brew install opencv
```

### 步骤2：从源码编译 librealsense
```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=true -DBUILD_PYTHON_BINDINGS=false
make -j4
sudo make install
```

### 步骤3：编译提取工具
```bash
cd /Users/macbookair/Documents/trae_projects/llm

g++ -std=c++11 \
    src/data_processing/extract_bag_frames.cpp \
    -lrealsense2 \
    $(pkg-config --cflags --libs opencv4) \
    -o extract_bag_frames
```

### 步骤4：运行
```bash
# 处理单个文件
./extract_bag_frames data/raw/25-10-08-20-35-40_0.bag data/frames 5

# 或批量处理
for bag in data/raw/*.bag; do
    ./extract_bag_frames "$bag" data/frames 5
done
```

---

## 方法3：使用 Docker（最稳定）

如果上述方法遇到问题，可以使用 Docker：

```bash
# 拉取包含 librealsense 的镜像
docker pull intelrealsense/librealsense:latest

# 运行容器并挂载数据目录
docker run -v $(pwd)/data:/data -it intelrealsense/librealsense:latest bash

# 在容器内编译和运行提取工具
```

---

## 验证安装

```bash
# 检查 librealsense
pkg-config --modversion realsense2

# 检查 OpenCV
pkg-config --modversion opencv4

# 测试 RealSense 工具
rs-enumerate-devices
```

---

## 预期输出

成功运行后，你将看到：
```
data/frames/
├── rgb/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
└── depth/
    ├── frame_000000.png
    ├── frame_000001.png
    └── ...
```

---

## 故障排除

### 问题1：找不到 realsense2
```bash
# 设置库路径
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### 问题2：OpenCV 版本不匹配
```bash
# 使用 opencv4 而不是 opencv
pkg-config --cflags --libs opencv4
```

### 问题3：权限问题
```bash
# 确保有执行权限
chmod +x scripts/compile_and_extract.sh
chmod +x extract_bag_frames
```

---

## 下一步

提取完成后：
1. 运行数据分析：`python src/data_processing/analyze_data.py data/frames/rgb --output docs`
2. 精选50-100张图像进行标注
3. 使用 LabelImg 标注障碍物
