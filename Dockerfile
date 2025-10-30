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

# Renderの $PORT を使って起動（デフォルト8000）
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
