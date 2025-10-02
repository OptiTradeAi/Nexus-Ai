#!/bin/bash
# Script para iniciar ambiente de desenvolvimento do Nexus AI

set -e

echo "🚀 Iniciando ambiente de desenvolvimento Nexus AI..."

# Verificar dependências
echo "📦 Verificando dependências..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.11+"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js não encontrado. Instale Node.js 18+"
    exit 1
fi

# Instalar dependências Python
echo "📦 Instalando dependências Python..."
cd server
pip install -r requirements.txt
cd ..

# Verificar se extensão existe
if [ ! -d "extension" ]; then
    echo "❌ Diretório da extensão não encontrado"
    exit 1
fi

# Iniciar servidor em background
echo "🖥️ Iniciando servidor..."
cd server
python server.py &
SERVER_PID=$!
cd ..

# Aguardar servidor inicializar
echo "⏳ Aguardando servidor inicializar..."
sleep 5

# Verificar se servidor está rodando
if curl -s http://localhost:9000/health > /dev/null; then
    echo "✅ Servidor iniciado com sucesso"
else
    echo "❌ Falha ao iniciar servidor"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Iniciar servidor web para interface
echo "🌐 Iniciando servidor web..."
cd web
python -m http.server 8000 &
WEB_PID=$!
cd ..

# Aguardar servidor web
sleep 3

echo "✅ Ambiente de desenvolvimento iniciado!"
echo ""
echo "📋 Informações:"
echo "   🖥️  Servidor API: http://localhost:9000"
echo "   🌐 Interface Web: http://localhost:8000"
echo "   📊 Health Check: http://localhost:9000/health"
echo ""
echo "📝 Próximos passos:"
echo "   1. Instale a extensão no Chrome/Edge:"
echo "      - Abra chrome://extensions/"
echo "      - Ative 'Modo desenvolvedor'"
echo "      - Clique 'Carregar sem compactação'"
echo "      - Selecione a pasta 'extension/'"
echo ""
echo "   2. Abra a interface de comparação:"
echo "      - http://localhost:8000/chart_comparacao.html"
echo ""
echo "   3. Para testes de screen-share:"
echo "      - http://localhost:8000/screen_sharer.html"
echo ""
echo "🛑 Para parar os serviços:"
echo "   kill $SERVER_PID $WEB_PID"

# Salvar PIDs para cleanup
echo $SERVER_PID > .server.pid
echo $WEB_PID > .web.pid

echo ""
echo "🎯 Ambiente pronto para desenvolvimento!"

