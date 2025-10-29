# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 学習時のクラス名 → 標準ラベル
CLASS_MAP = {
    'bottle-water-500ml': 'water_500ml',
    'bottle-water': 'water_2l'
}

# 標準ラベルごとの水容量（L）
VOLUME = {
    'water_500ml': 0.5,
    'water_2l': 2.0
}

def detect_bottles(model: YOLO, img_path: str, conf_thresh: float = 0.5) -> List[dict]:
    results = model(img_path)
    items = []
    names = model.names
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            if conf < conf_thresh:
                continue
            cls_id = int(cls_id)
            original_class = names[cls_id]
            std_class = CLASS_MAP.get(original_class)
            if std_class is None:
                continue
            items.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'class': std_class,
                'conf': float(conf)
            })
    return items

def estimate_days(items: list, family_size: int) -> Tuple[Dict[str,int], float, float]:
    counts = {'water_500ml': 0, 'water_2l': 0}
    total_l = 0.0
    for it in items:
        cls = it['class']
        counts[cls] += 1
        total_l += VOLUME[cls]
    daily_need = max(int(family_size), 1) * 3.0  # 1人1日=3L
    days = total_l / daily_need if daily_need > 0 else 0.0
    return counts, total_l, days

def _get_font():
    try:
        return ImageFont.truetype('arial.ttf', 18)
    except Exception:
        return ImageFont.load_default()

def visualize(img_path: str, items: list, counts: dict,
              total_l: float, days: float, family_size: int, out_path: str) -> str:
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = _get_font()

    for it in items:
        x1, y1, x2, y2 = it['bbox']
        label = f"{it['class']} ({it['conf']:.2f})"
        tw, th = draw.textlength(label, font=font), 18
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill='white')
        draw.text((x1 + 3, y1 - th - 4), label, fill='red', font=font)

    summary = f"{family_size} persons: 500ml x {counts['water_500ml']}, 2L x {counts['water_2l']} → {days:.1f} days (Total: {total_l:.1f} L)"
    w, h = img.size
    sw = draw.textlength(summary, font=font)
    sh = 18
    draw.rectangle([0, h - sh - 12, sw + 12, h], fill='white')
    draw.text((6, h - sh - 8), summary, fill='black', font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path

def analyze_photo_file(model: YOLO, img_path: str, persons: int, conf_thresh: float, out_path: str) -> dict:
    items = detect_bottles(model, img_path, conf_thresh=conf_thresh)
    counts, total_l, days = estimate_days(items, persons)
    vis_path = visualize(img_path, items, counts, total_l, days, persons, out_path)
    return {
        "counts": counts,
        "total_l": round(total_l, 1),
        "persons": persons,
        "estimated_days": round(days, 1),
        "visual_path": vis_path
    }
