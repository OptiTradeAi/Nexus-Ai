# Dockerfile — Nexus AI Server com Whisper, Tesseract e FFmpeg

# Base
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Atualiza pip
RUN pip install --upgrade pip setuptools wheel

# Instala PyTorch CPU
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch --no-cache-dir

# Instala pacotes Python necessários
RUN pip install --no-cache-dir openai-whisper pillow pytesseract aiohttp aiohttp-cors

# Copia o código Python
COPY . /app

# Copia a pasta da UI para o caminho esperado pelo servidor
# ATENÇÃO: substitua "web/ui/public" pelo caminho real da sua pasta com index.html
COPY web/ui/public /web/ui/public

# Expõe a porta usada pelo servidor
EXPOSE 10000

# Comando para iniciar o servidor
CMD ["python", "server.py"]
