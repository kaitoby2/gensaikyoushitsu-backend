# ===== Runtime only (軽量・フリープラン向け) =====

FROM python:3.11-slim

# ---- 環境変数 ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_INVENTORY=1 \
    STATIC_DIR=/app/static \
    WATER_BOTTLE_MODEL=/app/models/best.onnx

WORKDIR /app

# ---- OS 依存（opencv/onnxruntime用に最小限）----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python 依存 ----
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- アプリ本体 ----
COPY . ./

# 静的ファイル保存先（推論結果の画像を書き出す場所）
RUN mkdir -p "$STATIC_DIR/results"

# ---- ONNX モデル取得（必要なら --build-arg で差し替え）----
# 例: docker build --build-arg ONNX_URL=... --build-arg ALT_ONNX_URL=...
ARG ONNX_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.onnx"
ARG ALT_ONNX_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.onnx"

RUN set -eux; \
    mkdir -p /app/models; \
    (curl -fSL --retry 3 --connect-timeout 10 "$ONNX_URL" -o /app/models/best.onnx \
     || curl -fSL --retry 3 --connect-timeout 10 "$ALT_ONNX_URL" -o /app/models/best.onnx); \
    ls -lh /app/models; \
    # 1MB未満なら壊れたダウンロードとみなして失敗
    size="$(stat -c%s /app/models/best.onnx)"; \
    [ "$size" -ge 1000000 ]

# ---- ネットワーク公開 ----
EXPOSE 8000

# ---- 起動 ----
# Render などでは $PORT が注入される。未設定なら 8000 を使う
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
