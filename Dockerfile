FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 必要パッケージ（curl を追加）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# モデルをGitHub ReleasesからDL
RUN mkdir -p /app/models \
 && curl -L "https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v1.0.0/best.pt" \
 -o /app/models/best.pt

# Render が検知するように明示
EXPOSE 8000

# --- 必要なら curl を入れる（未導入なら入れてOK） ---
RUN apt-get update && apt-get install -y curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# --- GitHub Release の best.pt をビルド時に取り込む ---
ARG MODEL_URL="https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v.1.0.0/best.pt"
RUN mkdir -p /app/models \
    && curl -L "$MODEL_URL" -o /app/models/best.pt \
    && ls -lh /app/models

# 本番で参照する既定パス（後でENVで上書き不要にするため）
ENV WATER_BOTTLE_MODEL=/app/models/best.pt

# Renderの $PORT を使って起動（デフォルト8000）
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
