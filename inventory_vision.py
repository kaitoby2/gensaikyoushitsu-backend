# inventory_vision.py  — Freeプラン向け軽量版（ONNXRuntime使用）
import os
import asyncio
import gc
from io import BytesIO

from PIL import Image
import numpy as np
from ultralytics import YOLO  # .onnx を読ませると onnxruntime が使われます

# 画像縮小の最大長辺（環境変数で変更可）
MAX_SIDE = int(os.environ.get("INVENTORY_MAX_SIDE", "640"))
CONF     = float(os.environ.get("INVENTORY_CONF", "0.25"))

# グローバルにモデルを1回だけロード
_MODEL_SINGLETON = None
# Free(512MB)でのメモリ急増を避けるため、同時実行は1本に制限
_LOCK = asyncio.Lock()


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """画像をRGB化＋長辺MAX_SIDEに縮小してnumpy配列で返す"""
    im = Image.open(BytesIO(image_bytes)).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))  # アスペクト比保持で縮小
    return np.array(im)


def _get_model(model_path: str):
    """ONNXモデルをlazyにロード（.onnxを指定するとonnxruntimeで推論）"""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = YOLO(model_path)
    return _MODEL_SINGLETON


def _count_bottles(result) -> dict:
    """
    学習時のクラス割当てに合わせて本数を数える。
    例: class 0 = 500ml, class 1 = 2L （必要に応じて調整）
    """
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return {"n500": 0, "n2l": 0}

    cls = boxes.cls.cpu().numpy().astype(int)
    n500 = int((cls == 0).sum())
    n2l  = int((cls == 1).sum())
    return {"n500": n500, "n2l": n2l}


async def analyze_bottles(image_bytes: bytes, model_path: str) -> dict:
    """
    画像バイト列を受け取り、500ml/2Lの推定本数を返す。
    返り値は {'n500': int, 'n2l': int}
    """
    img = _preprocess(image_bytes)
    model = _get_model(model_path)

    # 1リクエストずつ処理してピークメモリを抑える
    async with _LOCK:
        results = model.predict(
            source=img,
            imgsz=MAX_SIDE,
            device="cpu",
            conf=CONF,
            verbose=False,
        )

    out = _count_bottles(results[0])

    # 明示的に解放を促す
    del img, results
    gc.collect()

    return out
