# ---------- Stage 1: exporter (best.pt -> best.onnx) ----------
FROM python:3.11-slim AS exporter
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# best.pt を GitHub Release から取得
# ★重要：あなたのリリースの「Download」ボタンのURLを下記 MODEL_URL に貼ってください
#   例）https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v1.0.0/best.pt
#   例）https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.pt
ARG MODEL_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v1.0.0/best.pt"
ARG ALT_MODEL_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.pt"

RUN mkdir -p /tmp/models \
 && (curl -fL "$MODEL_URL" -o /tmp/models/best.pt || curl -fL "$ALT_MODEL_URL" -o /tmp/models/best.pt)

# 変換にのみ Torch を使用（runtime には含めない）
RUN pip install --no-cache-dir ultralytics==8.2.103 \
    torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cpu

# ONNX へエクスポート（出力: /tmp/models/best.onnx）
RUN python - <<'PY'
from ultralytics import YOLO
p="/tmp/models"
m=YOLO(f"{p}/best.pt")
onnx = m.export(format="onnx", dynamic=True, simplify=True, opset=12,
                imgsz=640, project=p, name="onnx", exist_ok=True)
print("EXPORTED:", onnx)
import shutil, glob; shutil.copyfile(glob.glob(f"{p}/onnx/*.onnx")[0], f"{p}/best.onnx")
PY
RUN ls -lh /tmp/models

# ---------- Stage 2: runtime (超軽量、本番実行用：Torchなし) ----------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    UVICORN_WORKERS=1
WORKDIR /app

# 依存は軽量版 requirements.txt（Torch/torchvision なし）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . .

# 変換済みONNXだけ取り込む
RUN mkdir -p /app/models
COPY --from=exporter /tmp/models/best.onnx /app/models/best.onnx

# Render の $PORT を使って起動
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers
