# Dockerfile — Corrigido
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU wheel
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch --no-cache-dir

# Install whisper and other python deps
RUN pip install --no-cache-dir openai-whisper pillow pytesseract aiohttp aiohttp-cors

# Copia todo o projeto para /app. 
# Isto inclui server.py em /app e a UI em /app/web/ui/public
COPY . /app

EXPOSE 10000

CMD ["python", "server.py"]
