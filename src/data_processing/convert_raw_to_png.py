import os
import sys
import numpy as np
import cv2
import glob
from tqdm import tqdm

def load_metadata(output_dir):
    meta_file = os.path.join(output_dir, "metadata.txt")
    if not os.path.exists(meta_file):
        print(f"错误: 找不到元数据文件 {meta_file}")
        return None
        
    meta = {}
    with open(meta_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=')
                meta[key] = val
    return meta

def convert_raw_to_png(output_dir):
    print(f"正在转换 Raw 数据: {output_dir}")
    meta = load_metadata(output_dir)
    if meta is None:
        return
    
    rgb_w = int(meta['rgb_width'])
    rgb_h = int(meta['rgb_height'])
    depth_w = int(meta['depth_width'])
    depth_h = int(meta['depth_height'])
    
    rgb_raw_dir = os.path.join(output_dir, "rgb_raw")
    depth_raw_dir = os.path.join(output_dir, "depth_raw")
    
    rgb_out_dir = os.path.join(output_dir, "rgb")
    depth_out_dir = os.path.join(output_dir, "depth")
    
    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(depth_out_dir, exist_ok=True)
    
    # Convert RGB
    rgb_files = sorted(glob.glob(os.path.join(rgb_raw_dir, "*.raw")))
    print(f"发现 {len(rgb_files)} 个 RGB 帧")
    
    if rgb_files:
        for f in tqdm(rgb_files, desc="Converting RGB"):
            try:
                # Read raw
                raw_data = np.fromfile(f, dtype=np.uint8)
                # Reshape
                if raw_data.size != rgb_h * rgb_w * 3:
                    print(f"警告: 文件大小不匹配 {f}")
                    continue
                img = raw_data.reshape((rgb_h, rgb_w, 3))
                # RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # Save
                basename = os.path.basename(f).replace(".raw", ".jpg")
                cv2.imwrite(os.path.join(rgb_out_dir, basename), img_bgr)
            except Exception as e:
                print(f"处理 {f} 时出错: {e}")
        
    # Convert Depth
    depth_files = sorted(glob.glob(os.path.join(depth_raw_dir, "*.raw")))
    print(f"发现 {len(depth_files)} 个 Depth 帧")
    
    if depth_files:
        for f in tqdm(depth_files, desc="Converting Depth"):
            try:
                # Read raw (16-bit)
                raw_data = np.fromfile(f, dtype=np.uint16)
                # Reshape
                if raw_data.size != depth_h * depth_w:
                    print(f"警告: 文件大小不匹配 {f}")
                    continue
                img = raw_data.reshape((depth_h, depth_w))
                # Save as PNG (16-bit preserved)
                basename = os.path.basename(f).replace(".raw", ".png")
                cv2.imwrite(os.path.join(depth_out_dir, basename), img)
            except Exception as e:
                print(f"处理 {f} 时出错: {e}")
        
    print("转换完成！")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_raw_to_png.py <output_dir>")
        sys.exit(1)
        
    convert_raw_to_png(sys.argv[1])
