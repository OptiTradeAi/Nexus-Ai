# ⚡ Início Rápido - Nexus AI

## 🚀 Setup em 5 Minutos

### 1. Pré-requisitos
- Python 3.11+
- Chrome ou Edge
- Git

### 2. Instalação
```bash
# Clone o repositório
git clone <repository-url>
cd nexus_ai

# Inicie o ambiente de desenvolvimento
./scripts/start_dev.sh
```

### 3. Instalar Extensão
1. Abra Chrome/Edge
2. Vá para `chrome://extensions/`
3. Ative "Modo desenvolvedor"
4. Clique "Carregar sem compactação"
5. Selecione a pasta `extension/`

### 4. Testar Sistema
1. Abra http://localhost:8000/chart_comparacao.html
2. Abra a HomeBroker em outra aba
3. Verifique se dados aparecem no gráfico

## 🧪 Executar Testes
```bash
./scripts/run_tests.sh
```

## 🛑 Parar Serviços
```bash
./scripts/stop_dev.sh
```

## 📊 URLs Importantes
- **API Server**: http://localhost:9000
- **Interface Web**: http://localhost:8000
- **Health Check**: http://localhost:9000/health
- **Comparação**: http://localhost:8000/chart_comparacao.html
- **Screen Share**: http://localhost:8000/screen_sharer.html

## ❓ Problemas Comuns

**Extensão não funciona:**
- Verifique se está na página da HomeBroker
- Recarregue a extensão
- Verifique console do navegador

**Servidor não inicia:**
- Verifique se porta 9000 está livre
- Instale dependências: `pip install -r server/requirements.txt`

**Interface não carrega:**
- Verifique se porta 8000 está livre
- Verifique se servidor está rodando

## 📚 Documentação Completa
- [README.md](README.md) - Documentação completa
- [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) - Plano de execução
- [docs/HOW_TO_TEST.md](docs/HOW_TO_TEST.md) - Guia de testes

