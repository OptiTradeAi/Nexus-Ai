#!/bin/bash
# Script para parar ambiente de desenvolvimento do Nexus AI

echo "🛑 Parando ambiente de desenvolvimento Nexus AI..."

# Parar servidor se PID existe
if [ -f ".server.pid" ]; then
    SERVER_PID=$(cat .server.pid)
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "🖥️ Parando servidor (PID: $SERVER_PID)..."
        kill $SERVER_PID
        echo "✅ Servidor parado"
    else
        echo "⚠️ Servidor já estava parado"
    fi
    rm .server.pid
fi

# Parar servidor web se PID existe
if [ -f ".web.pid" ]; then
    WEB_PID=$(cat .web.pid)
    if kill -0 $WEB_PID 2>/dev/null; then
        echo "🌐 Parando servidor web (PID: $WEB_PID)..."
        kill $WEB_PID
        echo "✅ Servidor web parado"
    else
        echo "⚠️ Servidor web já estava parado"
    fi
    rm .web.pid
fi

# Cleanup adicional - matar processos por porta
echo "🧹 Limpeza adicional..."

# Matar processos na porta 9000 (servidor)
SERVER_PIDS=$(lsof -ti:9000 2>/dev/null || true)
if [ ! -z "$SERVER_PIDS" ]; then
    echo "🔧 Matando processos na porta 9000..."
    echo $SERVER_PIDS | xargs kill -9 2>/dev/null || true
fi

# Matar processos na porta 8000 (web)
WEB_PIDS=$(lsof -ti:8000 2>/dev/null || true)
if [ ! -z "$WEB_PIDS" ]; then
    echo "🔧 Matando processos na porta 8000..."
    echo $WEB_PIDS | xargs kill -9 2>/dev/null || true
fi

echo "✅ Ambiente de desenvolvimento parado!"

