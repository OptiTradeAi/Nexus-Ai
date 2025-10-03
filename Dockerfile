# Dockerfile — with whisper (CPU), tesseract and ffmpeg
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

# 1. COPIA OS ARQUIVOS DO SERVIDOR (agora que o Dockerfile está na raiz)
# Copia o conteúdo da pasta ./server/ (que contém server.py) para o /app
COPY ./server/ /app

# 2. CORREÇÃO DO ERRO 404: COPIA A UI para o local esperado pelo servidor
# O caminho de origem é ./web/ui/public (a partir da raiz)
# O caminho de destino é /web/ui/public (dentro do container)
COPY ./web/ui/public /web/ui/public 

EXPOSE 10000

# O comando CMD permanece o mesmo, pois server.py está em /app
CMD ["python", "server.py"]
