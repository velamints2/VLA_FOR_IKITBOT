#!/usr/bin/env python3
"""
通过 API 为 Label Studio 创建项目并导入 200 张图像
"""
import requests
import time
import json
from pathlib import Path
import sys

LS_URL = 'http://localhost:8080'
LS_API_TOKEN = 'devtoken123'
PROJECT_TITLE = 'Obstacle Detection'
IMAGES_DIR = Path('data/seed_dataset_v2')

headers = {'Authorization': f'Token {LS_API_TOKEN}', 'Content-Type': 'application/json'}

def wait_for_service(max_attempts=30):
    """等待 Label Studio 启动"""
    print("等待 Label Studio 启动...", end='', flush=True)
    for i in range(max_attempts):
        try:
            r = requests.get(f'{LS_URL}/', timeout=2)
            print(" ✓")
            return True
        except:
            time.sleep(1)
            print(".", end='', flush=True)
    print(" ✗")
    return False

def create_project():
    """创建或获取项目"""
    # 尝试获取现有项目
    try:
        r = requests.get(f'{LS_URL}/api/projects', headers=headers, timeout=5)
        if r.status_code == 200:
            projects = r.json().get('results', [])
            for p in projects:
                if p.get('title') == PROJECT_TITLE:
                    print(f"✓ 项目已存在: ID={p['id']}")
                    return p['id']
    except Exception as e:
        print(f"✗ 获取项目列表失败: {e}")
        return None
    
    # 创建新项目
    print("创建新项目...", end='', flush=True)
    try:
        label_config = Path('label_studio/config.xml').read_text()
        payload = {
            'title': PROJECT_TITLE,
            'label_config': label_config
        }
        r = requests.post(f'{LS_URL}/api/projects', headers=headers, json=payload, timeout=10)
        if r.status_code in (200, 201):
            project_id = r.json()['id']
            print(f" ✓ (ID={project_id})")
            return project_id
        else:
            print(f" ✗ ({r.status_code})")
            print(f"  错误: {r.text[:200]}")
            return None
    except Exception as e:
        print(f" ✗ ({e})")
        return None

def import_images(project_id):
    """导入图像"""
    images = sorted([p for p in IMAGES_DIR.glob('*.jpg')] + [p for p in IMAGES_DIR.glob('*.png')])
    print(f"\n发现 {len(images)} 张图像")
    
    if not images:
        print("✗ 没有找到图像！")
        return 0
    
    print("导入图像...", flush=True)
    created = 0
    failed = 0
    
    for idx, img_path in enumerate(images):
        # 使用相对路径，Label Studio 会从 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT 加载
        rel_path = str(img_path)
        task_data = {'data': {'image': rel_path}}
        
        try:
            r = requests.post(
                f'{LS_URL}/api/projects/{project_id}/tasks',
                headers=headers,
                json=task_data,
                timeout=5
            )
            if r.status_code in (200, 201):
                created += 1
            else:
                failed += 1
                if idx < 3:  # 只打印前几个错误
                    print(f"  [错误] {img_path.name}: {r.status_code}")
        except Exception as e:
            failed += 1
            if idx < 3:
                print(f"  [错误] {img_path.name}: {e}")
        
        # 进度显示
        if (idx + 1) % 50 == 0 or idx == len(images) - 1:
            pct = int(100 * (idx + 1) / len(images))
            print(f"  [{pct:3d}%] 已导入 {idx + 1}/{len(images)} ✓ {created} ✗ {failed}")
    
    print(f"\n✓ 导入完成: {created}/{len(images)} 成功，{failed} 失败")
    return created

def main():
    print("=" * 70)
    print("Label Studio 项目初始化")
    print("=" * 70)
    
    if not wait_for_service():
        print("\n✗ 无法连接到 Label Studio，请检查服务是否运行")
        sys.exit(1)
    
    project_id = create_project()
    if not project_id:
        print("\n✗ 创建项目失败")
        sys.exit(1)
    
    count = import_images(project_id)
    
    print("\n" + "=" * 70)
    print("✓ 初始化完成！")
    print("=" * 70)
    print(f"\n在浏览器中打开: {LS_URL}")
    print("  1. 登录到 Label Studio")
    print(f"  2. 打开 '{PROJECT_TITLE}' 项目")
    print("  3. 开始标注 200 张图像")
    print("\n完成后导出为 JSON 格式")

if __name__ == '__main__':
    main()
