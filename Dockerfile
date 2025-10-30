# ベースイメージ
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 依存ライブラリのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコピー
COPY . .

# ✅ モデルファイルをGitHub Releasesからダウンロード
RUN mkdir -p /app/models \
    && curl -L "https://github.com/kaitoby2/gensaikyoushitsu-backend/releases/download/v1.0.0/best.pt" \
    -o /app/models/best.pt

# ✅ コンテナ起動コマンド（最後に記述）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
