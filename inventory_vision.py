# inventory_vision.py  —— PyTorch/YOLOv8版 (元のロジックを回復)
# -*- coding: utf-8 -*-
from pathlib import Path
import os
from typing import Tuple, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO # ★ PyTorch/YOLOv8を再導入

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

# フォントロード（Render環境のフォントパスに依存。適宜調整が必要な場合あり）
_FONT_PATH = os.environ.get("FONT_PATH") or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT: Optional[ImageFont.ImageFont] = None

def _get_font(size=14):
    """フォントを遅延ロード"""
    global _FONT
    if _FONT is None:
        try:
            # Render環境で利用可能なフォントパスを使用
            _FONT = ImageFont.truetype(_FONT_PATH, size)
        except IOError:
            print(f"Warning: Font not found at {_FONT_PATH}. Using default font.")
            _FONT = ImageFont.load_default()
    return _FONT

def detect_bottles(model: YOLO, img_path: str, conf_thresh: float = 0.5) -> List[dict]:
    """YOLOv8モデルで画像を推論し、検出アイテムのリストを返す"""
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
                'conf': float(conf),
            })
    return items

def annotate_and_save(img_path: str, items: List[dict], counts: Dict[str, int], total_l: float, days: float, family_size: int, out_path: str) -> str:
    """検出結果を画像に描画し、ファイルに保存する"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = _get_font()

    for it in items:
        x1, y1, x2, y2 = it['bbox']
        label = f"{it['class']} ({it['conf']:.2f})"
        
        try: tw = draw.textlength(label, font=font)
        except AttributeError: tw = len(label) * 8
        th = 18
        
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill='white')
        draw.text((x1 + 3, y1 - th - 4), label, fill='red', font=font)

    summary = f"{family_size} persons: 500ml x {counts['water_500ml']}, 2L x {counts['water_2l']} → {days:.1f} days (Total: {total_l:.1f} L)"
    w, h = img.size
    
    try: sw = draw.textlength(summary, font=font)
    except AttributeError: sw = len(summary) * 8
        
    sh = 18
    draw.rectangle([0, h - sh - 12, sw + 12, h], fill='white')
    draw.text((6, h - sh - 8), summary, fill='black', font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format='JPEG', quality=95)
    return out_path

def analyze_photo_file(model: YOLO, img_path: str, persons: int, conf_thresh: float, out_path: str) -> Dict:
    """YOLOv8推論と結果の集計を行うメイン関数"""
    
    items = detect_bottles(model, img_path, conf_thresh)
    
    # クラスごとのカウント
    counts: Dict[str, int] = {}
    for it in items:
        counts[it['class']] = counts.get(it['class'], 0) + 1
        
    n500 = counts.get('water_500ml', 0)
    n2l = counts.get('water_2l', 0)

    # 合計容量計算
    water_from_image_l = n500 * VOLUME['water_500ml'] + n2l * VOLUME['water_2l']
    
    # 必要な日数を計算 (main.pyのデフォルト値を使用して概算)
    DAILY_NEED_PER_PERSON_L = 3.0
    need_water_per_day = DAILY_NEED_PER_PERSON_L * max(1, persons)
    estimated_days = round((water_from_image_l / need_water_per_day), 1) if need_water_per_day > 0 else 0.0

    # 画像に描画して保存
    visual_path = annotate_and_save(
        img_path, items, 
        {'water_500ml': n500, 'water_2l': n2l}, 
        water_from_image_l, 
        estimated_days, 
        persons, 
        out_path
    )

    return {
        "persons": persons,
        "counts": {'water_500ml': n500, 'water_2l': n2l},
        "total_l": water_from_image_l,
        "estimated_days": estimated_days,
        "visual_path": visual_path,
    }
