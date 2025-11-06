# ===== Runtime only (軽量・フリープラン向け) =====
FROM python:3.11-slim

# ---- 環境変数（デフォルト）----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_INVENTORY=1 \
    # ← データ既定パス。Render の Persistent Disk を使うなら DATA_DIR=/data に上書き
    DATA_DIR=/app/data \
    STATIC_DIR=/app/static \
    WATER_BOTTLE_MODEL=/app/models/best.onnx

WORKDIR /app

# ---- OS 依存（opencv/onnxruntime用に最小限）----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python 依存 ----
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- アプリ本体 ----
# *.py / templates / static / など一式
COPY . ./

# ---- data を確実に含める（.dockerignore で誤って除外されていても検知）----
# 1) 既定の保存先を用意
RUN mkdir -p "$DATA_DIR" "$STATIC_DIR/results" /app/models

# 2) リポジトリ内の data/ をコンテナにコピー
#    ※ .dockerignore に data/ が入っていると COPY 時に落ちるので、その場合は build で失敗させて気付けるようにする
COPY ./data/ ${DATA_DIR}/

# 3) 必須ファイルの存在チェック（無ければビルド失敗→原因が一目瞭然）
RUN test -f "${DATA_DIR}/scenarios_gifu_gotanda.json" \
 && test -f "${DATA_DIR}/quiz_v1.csv" \
 && test -f "${DATA_DIR}/advice_rules_v1.json" \
 && test -f "${DATA_DIR}/stock_baseline_v1.json"

# ---- ONNX モデル取得（必要なら）----
ARG ONNX_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.onnx"
ARG ALT_ONNX_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.onnx"
RUN set -eux; \
    (curl -fSL --retry 3 --connect-timeout 10 "$ONNX_URL" -o /app/models/best.onnx \
     || curl -fSL --retry 3 --connect-timeout 10 "$ALT_ONNX_URL" -o /app/models/best.onnx); \
    size="$(stat -c%s /app/models/best.onnx 2>/dev/null || echo 0)"; \
    [ "$size" -ge 1000000 ] || { echo "ONNX model missing or too small; inventory will fail"; exit 1; }

# ---- ヘルスチェック（起動後に /health が返るか）----
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD \
  curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# ---- ネットワーク公開 ----
EXPOSE 8000

# ---- 起動 ----
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
