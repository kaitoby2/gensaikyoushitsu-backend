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
      curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python依存（requirements.txt に onnxruntime / opencv-python-headless / fastapi / uvicorn 等が入っている想定）----
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- アプリ本体 ----
COPY . ./

# 静的ファイル保存先（推論結果の画像を書き出す場所）
RUN mkdir -p ${STATIC_DIR}/results

# ---- ONNX モデル取得（GitHub Releases の best.onnx 直リンク）----
# 必要に応じてビルド引数で差し替え可能: --build-arg ONNX_URL=...
ARG ONNX_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.onnx"
# 代替URLがあればここに
ARG ALT_ONNX_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.onnx"

RUN mkdir -p /app/models \
 && (curl -fSL --retry 3 --connect-timeout 10 "$ONNX_URL" -o /app/models/best.onnx \
     || curl -fSL --retry 3 --connect-timeout 10 "$ALT_ONNX_URL" -o /app/models/best.onnx) \
 && ls -lh /app/models \
 # 9Bなどの壊れたダウンロードを即検知（1MB未満なら失敗させる）
 && python - <<'PY'
import os, sys
p="/app/models/best.onnx"
sz=os.path.getsize(p) if os.path.exists(p) else 0
print("best.onnx size:", sz, "bytes")
# 1MB 未満は明らかな異常とみなす（URL間違い/Private等）
sys.exit(0 if sz >= 1_000_000 else 2)
PY

# ---- 起動 ----
# Render は $PORT を注入する。未設定時は 8000 を使う
CMD bash -lc 'uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}'
