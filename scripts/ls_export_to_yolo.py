#!/usr/bin/env python3
"""
Convert Label Studio export JSON to YOLO label files.
- Input: Label Studio project export (JSON), with rectanglelabels results.
- Output: YOLO txt files (class x_center y_center width height), normalized [0,1].

Usage:
  python scripts/ls_export_to_yolo.py \
      --export label_studio/data/export/project-*.json \
      --images-dir data/seed_dataset_v2 \
      --labels-dir data/seed_dataset_v2/labels

Notes:
- Class mapping matches label_studio/config.xml with 6 classes.
- If an image has no annotations, no txt file is written by default (can be changed).
"""
import json
import argparse
import os
from pathlib import Path

# Label name -> YOLO class id mapping (must match training config expectations)
LABEL_TO_ID = {
    'wire': 0,
    'slipper': 1,
    'sock': 2,
    'cable': 3,
    'toy': 4,          # maps to 'small_toy' in configs/data.yaml names index 4
    'obstacle': 5
}


def to_yolo_from_percent(x: float, y: float, w: float, h: float):
    """Convert LS percentage box to YOLO normalized center format."""
    x_n = x / 100.0
    y_n = y / 100.0
    w_n = w / 100.0
    h_n = h / 100.0
    # convert top-left to center
    cx_n = x_n + w_n / 2.0
    cy_n = y_n + h_n / 2.0
    return cx_n, cy_n, w_n, h_n


def sanitize_name(path_value: str) -> str:
    """Derive image basename from LS data.image value."""
    # LS may store 'file:///abs/path/to/image.jpg' or '/abs/path.jpg' or URL
    p = path_value
    if p.startswith('file://'):
        p = p[len('file://'):]  # strip scheme
    return os.path.basename(p)


def parse_export(export_path: Path):
    with export_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    # LS export could be list of tasks or dict; normalize to list
    if isinstance(data, dict) and 'tasks' in data:
        tasks = data['tasks']
    elif isinstance(data, list):
        tasks = data
    else:
        tasks = []
    return tasks


def convert(export_path: Path, images_dir: Path, labels_dir: Path, write_empty: bool = False):
    tasks = parse_export(export_path)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for task in tasks:
        data = task.get('data', {})
        image_val = data.get('image') or data.get('image_url') or ''
        img_base = sanitize_name(image_val)
        if not img_base:
            skipped += 1
            continue
        stem = Path(img_base).stem
        out_txt = labels_dir / f"{stem}.txt"

        # Collect rectangles from annotations
        ann_lines = []
        anns = task.get('annotations') or []
        for ann in anns:
            results = ann.get('result') or []
            for r in results:
                if r.get('type') != 'rectanglelabels':
                    continue
                val = r.get('value') or {}
                x = float(val.get('x', 0.0))
                y = float(val.get('y', 0.0))
                w = float(val.get('width', 0.0))
                h = float(val.get('height', 0.0))
                labels = val.get('rectanglelabels') or []
                if not labels:
                    continue
                label_name = str(labels[0])
                if label_name not in LABEL_TO_ID:
                    # skip unknown label
                    continue
                cls_id = LABEL_TO_ID[label_name]
                cx, cy, wn, hn = to_yolo_from_percent(x, y, w, h)
                ann_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        if ann_lines or write_empty:
            with out_txt.open('w', encoding='utf-8') as f:
                for line in ann_lines:
                    f.write(line + '\n')
            written += 1
        else:
            skipped += 1

    return written, skipped


def main():
    ap = argparse.ArgumentParser(description='Convert Label Studio export JSON to YOLO labels')
    ap.add_argument('--export', required=True, help='Path to Label Studio export JSON')
    ap.add_argument('--images-dir', required=False, default='data/seed_dataset_v2', help='Images directory (optional)')
    ap.add_argument('--labels-dir', required=False, default='data/seed_dataset_v2/labels', help='Output labels directory')
    ap.add_argument('--write-empty', action='store_true', help='Write empty label files for images with no annotations')
    args = ap.parse_args()

    export_path = Path(args.export)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not export_path.exists():
        raise SystemExit(f"Export file not found: {export_path}")

    written, skipped = convert(export_path, images_dir, labels_dir, args.write_empty)
    print('=' * 60)
    print('Label Studio → YOLO 转换完成')
    print('- JSON:', export_path)
    print('- 输出标签目录:', labels_dir)
    print(f'- 已写入: {written} 文件, 跳过: {skipped}')
    print('=' * 60)


if __name__ == '__main__':
    main()
