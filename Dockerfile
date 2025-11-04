# ===== Runtime only (軽量) =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 画像診断を有効化（main.pyが読む）
    ENABLE_INVENTORY=1 \
    WATER_BOTTLE_MODEL=/app/models/best.onnx

WORKDIR /app

# OSに最低限のライブラリ（opencv/onnxruntime用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 依存を先に入れてキャッシュを効かせる
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# アプリ本体
COPY . ./

# Release から ONNX を直接ダウンロード（※ v1.0.0 に合わせてある）
RUN mkdir -p /app/models \
 && curl -L "https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v1.0.0/best.onnx" \
      -o /app/models/best.onnx \
 && ls -lh /app/models

# Render は $PORT を注入してくる
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
