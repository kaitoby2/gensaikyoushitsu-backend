# inventory_vision.py  —— ultralytics不要版（onnxruntime 直接推論）
import os
import io
import uuid
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
import onnxruntime as ort

# モデルパスの解決（RenderのENVで WATAR_BOTTLE_MODEL=/app/models/best.onnx にしてあればそれを使う）
MODEL_ONNX = (
    os.environ.get("MODEL_ONNX")
    or os.environ.get("WATER_BOTTLE_MODEL")
    or "/app/models/best.onnx"
)

# 出力先（/static/results に書く。main.py 側で STATIC_DIR を /app/static にしている想定）
STATIC_ROOT = Path(os.environ.get("STATIC_DIR", "/app/static")).resolve()
RESULTS_DIR = (STATIC_ROOT / "results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1本あたりの容量（必要に応じてENVで調整。デフォルト2.0L）
BOTTLE_LITERS = float(os.environ.get("BOTTLE_LITERS", "2.0"))

# 推論用セッション
_session: ort.InferenceSession = None
_input_name: str = ""
_input_h = 640
_input_w = 640


def _load_session():
    global _session, _input_name, _input_h, _input_w
    if _session is None:
        model_path = Path(MODEL_ONNX)
        
        if not model_path.exists() or model_path.stat().st_size == 0:
            print(f"[WARN] ONNX model not found or empty: {MODEL_ONNX}. Vision will be skipped.")
            # モデルファイルが見つからない場合は、セッションロードをスキップ
            return 
        
        print(f"[INFO] Loading ONNX model from {MODEL_ONNX}...")
        
        # ★★★【重要】CPU使用を制限し、メモリ消費を抑える（502対策）★★★
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1 # 1スレッドに制限
        sess_options.inter_op_num_threads = 1 # 1スレッドに制限
        
        _session = ort.InferenceSession(MODEL_ONNX, sess_options)
        
        # 入力情報の取得
        _input_name = _session.get_inputs()[0].name
        shape = _session.get_inputs()[0].shape
        _input_w = shape[2]
        _input_h = shape[3]
        print(f"[INFO] ONNX model loaded. Input: {_input_name} Shape: {_input_w}x{_input_h}")

def _letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)
               ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    # x = [cx, cy, w, h] -> [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th=0.45) -> List[int]:
    # boxes: [N,4] (xyxy), scores: [N]
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_th]
    return keep


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # box: [4], boxes: [M,4] in xyxy
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def analyze_bottles(file_bytes: bytes) -> Dict:
    """
    画像bytesを受け取り、水ボトルの推定本数とリットル数を返す。
    返却: {
    　"count": int,
    　"estimated_liters": float,
      "annotated_relpath": str | None
     }
    """
    _load_session()

    # モデルのロードに失敗した場合（_load_session内でモデルファイルが見つからなかった場合）は、ここで処理をスキップ
    if _session is None:
        print("[WARN] Vision skipped because ONNX model session is not loaded.")
        return {"count": 0, "estimated_liters": 0.0, "annotated_relpath": None}

    # 画像デコード（BGR）
    img_array = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes.")

    h0, w0 = img.shape[:2]

    # 前処理（レターボックス -> RGB -> CHW -> float32 / 255）
    lb, r, (dw, dh) = _letterbox(img, (_input_h, _input_w))
    im = lb[:, :, ::-1]  # BGR -> RGB
    im = im.transpose(2, 0, 1)  # HWC -> CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    im = im[None, :]  # [1,3,640,640]

    # 推論
    out = _session.run(None, {_input_name: im})[0]  # shape: [1,6,N] （1クラス想定）
    out = np.squeeze(out, axis=0)                   # [6,N]
    out = out.T                                     # [N,6] -> [cx,cy,w,h,conf,clsconf]

    if out.size == 0:
        return {"count": 0, "estimated_liters": 0.0, "annotated_relpath": None}

    # 信頼度（obj_conf * cls_conf）
    scores = out[:, 4] * out[:, 5]

    # スコアしきい値
    mask = scores >= float(os.environ.get("BOTTLE_SCORE_TH", "0.35"))
    out = out[mask]
    scores = scores[mask]
    if out.size == 0:
        return {"count": 0, "estimated_liters": 0.0, "annotated_relpath": None}

    # xywh(640基準) -> xyxy(レターボックス画像基準)
    boxes = _xywh2xyxy(out[:, :4])

    # レターボックス解除 -> 元画像座標へ
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r

    # 範囲クリップ
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)

    # NMS
    keep = _nms(boxes, scores, iou_th=float(os.environ.get("BOTTLE_NMS_IOU", "0.45")))
    boxes = boxes[keep]
    scores = scores[keep]

    count = int(len(boxes))
    est_liters = round(count * BOTTLE_LITERS, 2)

    # 簡易アノテーションを保存
    vis = img.copy()
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, (0, 200, 0), 2)
        cv2.putText(vis, f"bottle {sc:.2f}", (p1[0], max(0, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)

    fname = f"inv_{uuid.uuid4().hex[:8]}.jpg"
    out_path = RESULTS_DIR / fname
    cv2.imwrite(str(out_path), vis)

    # 最終的な結果を返す
    rel = out_path.relative_to(STATIC_ROOT).as_posix()
    return {"count": count, "estimated_liters": est_liters, "annotated_relpath": rel}
