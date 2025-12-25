#!/usr/bin/env python3
"""
更新 Label Studio 任务，使用 HTTP URL 提供的图像而不是相对路径
"""
import requests
import json
from pathlib import Path

LS_URL = 'http://localhost:8080'
LS_API_TOKEN = 'devtoken123'
PROJECT_ID = 2
IMAGES_DIR = Path('data/seed_dataset_v2')
IMAGE_SERVER_URL = 'http://localhost:8081'

headers = {'Authorization': f'Token {LS_API_TOKEN}', 'Content-Type': 'application/json'}

def update_tasks():
    """获取所有任务并更新图像 URL"""
    print("获取任务列表...", end='', flush=True)
    
    # 获取所有任务
    all_tasks = []
    offset = 0
    limit = 100
    
    while True:
        r = requests.get(
            f'{LS_URL}/api/projects/{PROJECT_ID}/tasks',
            headers=headers,
            params={'offset': offset, 'limit': limit}
        )
        if r.status_code != 200:
            print(f"\n✗ 获取失败: {r.status_code}")
            return False
        
        tasks = r.json()
        if not tasks:
            break
        
        all_tasks.extend(tasks)
        offset += len(tasks)
    
    print(f" ✓ ({len(all_tasks)} 个任务)")
    
    print("更新 URL...", flush=True)
    updated = 0
    failed = 0
    
    for idx, task in enumerate(all_tasks):
        try:
            # 提取文件名
            image_path = task.get('data', {}).get('image', '')
            if not image_path:
                continue
            
            filename = Path(image_path).name
            # 生成完整 URL
            new_url = f'{IMAGE_SERVER_URL}/{filename}'
            
            # 更新任务
            payload = {'data': {'image': new_url}}
            r = requests.patch(
                f'{LS_URL}/api/tasks/{task["id"]}',
                headers=headers,
                json=payload,
                timeout=5
            )
            
            if r.status_code in (200, 201):
                updated += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"  [错误] 任务 {task['id']}: {r.status_code}")
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  [错误] 任务 {task['id']}: {e}")
        
        if (idx + 1) % 50 == 0 or idx == len(all_tasks) - 1:
            pct = int(100 * (idx + 1) / len(all_tasks))
            print(f"  [{pct:3d}%] {idx + 1}/{len(all_tasks)} - ✓ {updated} ✗ {failed}")
    
    print(f"\n✓ 更新完成: {updated}/{len(all_tasks)} 成功")
    return True

if __name__ == '__main__':
    print("=" * 70)
    print("更新 Label Studio 任务 URL")
    print("=" * 70 + "\n")
    
    success = update_tasks()
    
    if success:
        print("\n" + "=" * 70)
        print("✓ 已完成！图像现在应该通过 HTTP 正常加载")
        print("=" * 70)
        print("\n在浏览器中打开: http://localhost:8080")
        print("然后开始标注任务")

