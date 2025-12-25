#!/usr/bin/env python3
"""
Import images and YOLO pre-annotations into Label Studio via API.
Requires: requests
Usage: .venv311/bin/python scripts/ls_import_with_preannots.py
"""
import os
import json
import requests
from glob import glob

LS_URL = os.getenv('LS_URL', 'http://localhost:8080')
API_TOKEN = os.getenv('LS_TOKEN', 'devtoken123')
PROJECT_TITLE = os.getenv('LS_PROJECT', 'Obstacle Detection')
IMAGES_DIR = os.getenv('LS_IMAGES_DIR', 'data/seed_dataset_v2')
PREANN_DIR = os.path.join(IMAGES_DIR, 'auto_labels')

headers = {'Authorization': f'Token {API_TOKEN}', 'Content-Type': 'application/json'}

def get_project_id():
    r = requests.get(f'{LS_URL}/api/projects', headers=headers)
    r.raise_for_status()
    projects = r.json().get('results', [])
    for p in projects:
        if p.get('title') == PROJECT_TITLE:
            return p.get('id')
    raise SystemExit('Project not found. Create it first via API or UI')

# mapping class index -> label name (must match config.xml)
LABEL_MAP = {
    '0': 'wire',
    '1': 'slipper',
    '2': 'sock',
    '3': 'cable',
    '4': 'toy',
    '5': 'obstacle'
}


def yolo_txt_to_ls_results(txt_path, img_w=None, img_h=None):
    results = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = parts[0:5]
            try:
                cx = float(cx); cy = float(cy); w = float(w); h = float(h)
            except:
                continue
            # YOLO are normalized [0,1] relative to image dims — convert to percentages
            x_pct = (cx - w/2.0) * 100.0
            y_pct = (cy - h/2.0) * 100.0
            width_pct = w * 100.0
            height_pct = h * 100.0
            label = LABEL_MAP.get(cls, f'class_{cls}')
            ann = {
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'x': x_pct,
                    'y': y_pct,
                    'width': width_pct,
                    'height': height_pct,
                    'rotation': 0,
                    'rectanglelabels': [label]
                }
            }
            results.append(ann)
    return results


def create_task(project_id, image_path, preann_path=None):
    abs_path = os.path.abspath(image_path)
    data = {'project': project_id, 'data': {'image': f'file://{abs_path}'}}
    r = requests.post(f'{LS_URL}/api/tasks', headers=headers, json=data)
    if r.status_code not in (200,201):
        print('Task create failed', r.status_code, r.text)
        return None
    task = r.json()
    task_id = task.get('id')
    # attach pre-annotations if exist
    if preann_path and os.path.exists(preann_path):
        results = yolo_txt_to_ls_results(preann_path)
        if results:
            ann_payload = {
                'result': results,
                'lead_time': 0.0,
                'was_cancelled': False,
                'ground_truth': False
            }
            ra = requests.post(f'{LS_URL}/api/tasks/{task_id}/annotations', headers=headers, json=ann_payload)
            if ra.status_code not in (200,201):
                print('Annotation upload failed for', task_id, ra.status_code, ra.text)
            else:
                print('Uploaded annotation for task', task_id)
    return task_id


def main():
    project_id = get_project_id()
    print('Project id:', project_id)
    imgs = sorted([p for p in glob(os.path.join(IMAGES_DIR, '*')) if p.lower().endswith(('.jpg','.jpeg','.png'))])
    print('Found images:', len(imgs))
    created = 0
    for p in imgs:
        base = os.path.basename(p)
        name, _ = os.path.splitext(base)
        pre = os.path.join(PREANN_DIR, f'{name}.txt')
        tid = create_task(project_id, p, preann_path=pre)
        if tid:
            created += 1
    print('Created tasks:', created)

if __name__ == '__main__':
    main()
