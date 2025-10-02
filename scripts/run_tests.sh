#!/bin/bash
# Script para executar testes do Nexus AI

set -e

echo "🧪 Executando testes do Nexus AI..."

# Verificar se pytest está instalado
if ! command -v pytest &> /dev/null; then
    echo "📦 Instalando pytest..."
    pip install pytest pytest-asyncio pytest-cov
fi

# Verificar se servidor está rodando
SERVER_RUNNING=false
if curl -s http://localhost:9000/health > /dev/null; then
    echo "✅ Servidor já está rodando"
    SERVER_RUNNING=true
else
    echo "🖥️ Iniciando servidor para testes..."
    cd server
    python server.py &
    SERVER_PID=$!
    cd ..
    sleep 5
    
    if curl -s http://localhost:9000/health > /dev/null; then
        echo "✅ Servidor iniciado para testes"
    else
        echo "❌ Falha ao iniciar servidor"
        exit 1
    fi
fi

# Verificar se interface web está rodando
WEB_RUNNING=false
if curl -s http://localhost:8000 > /dev/null; then
    echo "✅ Interface web já está rodando"
    WEB_RUNNING=true
else
    echo "🌐 Iniciando interface web para testes..."
    cd web
    python -m http.server 8000 &
    WEB_PID=$!
    cd ..
    sleep 3
fi

# Executar testes por categoria
echo ""
echo "🔧 Executando testes unitários..."
pytest tests/unit/ -v --tb=short

echo ""
echo "🔗 Executando testes de integração..."
pytest tests/integration/ -v --tb=short

echo ""
echo "⚡ Executando testes de performance..."
pytest tests/performance/ -v --tb=short --timeout=300

echo ""
echo "🌐 Executando testes E2E..."
pytest tests/e2e/ -v --tb=short --timeout=600

# Gerar relatório de cobertura
echo ""
echo "📊 Gerando relatório de cobertura..."
pytest tests/ --cov=server --cov=model --cov=tools --cov-report=html --cov-report=term

# Cleanup se iniciamos os serviços
if [ "$SERVER_RUNNING" = false ] && [ ! -z "$SERVER_PID" ]; then
    echo "🛑 Parando servidor de teste..."
    kill $SERVER_PID 2>/dev/null || true
fi

if [ "$WEB_RUNNING" = false ] && [ ! -z "$WEB_PID" ]; then
    echo "🛑 Parando interface web de teste..."
    kill $WEB_PID 2>/dev/null || true
fi

echo ""
echo "✅ Testes concluídos!"
echo "📊 Relatório de cobertura: htmlcov/index.html"

