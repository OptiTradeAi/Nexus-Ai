# Nexus AI Server - Dockerfile
# Imagem otimizada para deploy no Render

FROM python:3.11-slim

# Metadados
LABEL maintainer="Manus AI"
LABEL description="Nexus AI - Sistema de Trading com IA para Opções Binárias"
LABEL version="1.0.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=9000

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root
RUN useradd --create-home --shell /bin/bash nexus
WORKDIR /home/nexus/app

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY server.py .

# Mudar para usuário não-root
RUN chown -R nexus:nexus /home/nexus/app
USER nexus

# Expor porta
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Comando de inicialização
CMD ["python", "server.py"]

