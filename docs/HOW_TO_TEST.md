# 🧪 Guia de Testes - Nexus AI

**Versão:** 1.0.0  
**Autor:** Manus AI  
**Data:** 2025-09-29  

## 🎯 Objetivo

Este documento fornece um guia abrangente para testar todos os componentes do sistema Nexus AI, desde testes unitários até validação em produção. O guia está organizado por componente e tipo de teste, com scripts automatizados e procedimentos manuais detalhados.

## 📋 Visão Geral dos Testes

### Estratégia de Testes

O Nexus AI utiliza uma estratégia de testes em pirâmide, priorizando:

1. **Testes Unitários** (70%): Componentes individuais
2. **Testes de Integração** (20%): Interação entre componentes  
3. **Testes E2E** (10%): Fluxo completo do usuário

### Tipos de Testes

- **Funcionais**: Verificam se o sistema faz o que deveria fazer
- **Performance**: Validam latência, throughput e escalabilidade
- **Segurança**: Testam vulnerabilidades e proteção de dados
- **Usabilidade**: Avaliam experiência do usuário
- **Compatibilidade**: Verificam funcionamento em diferentes ambientes

## 🏗️ Configuração do Ambiente de Testes

### Pré-requisitos

```bash
# Dependências de desenvolvimento
pip install pytest pytest-asyncio pytest-cov
pip install selenium webdriver-manager
pip install locust  # Para testes de carga
npm install -g lighthouse  # Para testes de performance web
```

### Estrutura de Testes

```
tests/
├── unit/                   # Testes unitários
│   ├── test_server.py
│   ├── test_model.py
│   └── test_tools.py
├── integration/            # Testes de integração
│   ├── test_websocket.py
│   ├── test_extension.py
│   └── test_ocr.py
├── e2e/                   # Testes end-to-end
│   ├── test_full_flow.py
│   └── test_user_journey.py
├── performance/           # Testes de performance
│   ├── load_test.py
│   └── stress_test.py
├── fixtures/              # Dados de teste
│   ├── sample_ticks.json
│   └── sample_candles.json
└── conftest.py           # Configuração pytest
```

### Configuração Base

```python
# conftest.py
import pytest
import asyncio
import json
from pathlib import Path

@pytest.fixture
def sample_ticks():
    """Carrega ticks de exemplo para testes."""
    with open('tests/fixtures/sample_ticks.json') as f:
        return json.load(f)

@pytest.fixture
def test_server():
    """Inicia servidor de teste."""
    # Implementação do servidor de teste
    pass

@pytest.fixture
def browser():
    """Configura navegador para testes E2E."""
    from selenium import webdriver
    driver = webdriver.Chrome()
    yield driver
    driver.quit()
```

## 🔧 Testes do Servidor (server/)

### Testes Unitários

#### Teste de Agregação de Candles

```python
# tests/unit/test_server.py
import pytest
from server.server import CandleAggregator

class TestCandleAggregator:
    
    def test_candle_creation(self):
        """Testa criação de candle a partir de ticks."""
        aggregator = CandleAggregator('AAPL-OTC', '1m')
        
        # Adicionar ticks
        aggregator.add_tick(100.0, 1234567890)
        aggregator.add_tick(101.0, 1234567891)
        aggregator.add_tick(99.5, 1234567892)
        aggregator.add_tick(100.5, 1234567893)
        
        candle = aggregator.get_current_candle()
        
        assert candle['open'] == 100.0
        assert candle['high'] == 101.0
        assert candle['low'] == 99.5
        assert candle['close'] == 100.5
    
    def test_candle_timeframe(self):
        """Testa agregação por timeframe."""
        aggregator = CandleAggregator('AAPL-OTC', '1m')
        
        # Ticks no mesmo minuto
        base_time = 1234567800  # Início do minuto
        aggregator.add_tick(100.0, base_time)
        aggregator.add_tick(101.0, base_time + 30)
        
        # Tick no próximo minuto
        aggregator.add_tick(102.0, base_time + 60)
        
        # Deve ter criado novo candle
        assert len(aggregator.closed_candles) == 1
        assert aggregator.closed_candles[0]['close'] == 101.0
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast(self):
        """Testa broadcast de candles via WebSocket."""
        from server.server import broadcast_candle
        
        # Mock WebSocket connections
        mock_connections = []
        
        candle_data = {
            'symbol': 'AAPL-OTC',
            'timeframe': '1m',
            'open': 100.0,
            'high': 101.0,
            'low': 99.5,
            'close': 100.5
        }
        
        await broadcast_candle(candle_data, mock_connections)
        
        # Verificar se mensagem foi enviada
        # Implementar verificação baseada no mock
```

#### Teste de Endpoints HTTP

```python
import pytest
from aiohttp.test_utils import AioHTTPTestCase
from server.server import create_app

class TestHTTPEndpoints(AioHTTPTestCase):
    
    async def get_application(self):
        return create_app()
    
    async def test_health_endpoint(self):
        """Testa endpoint de health check."""
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200
        
        data = await resp.json()
        assert data['status'] == 'healthy'
        assert 'uptime' in data
        assert 'version' in data
    
    async def test_push_tick_endpoint(self):
        """Testa endpoint de recebimento de ticks."""
        tick_data = {
            'symbol': 'AAPL-OTC',
            'price': 100.50,
            'ts': 1234567890,
            'volume': 1
        }
        
        resp = await self.client.request(
            "POST", "/push",
            json=tick_data
        )
        
        assert resp.status == 200
        data = await resp.json()
        assert data['status'] == 'received'
    
    async def test_invalid_tick_data(self):
        """Testa validação de dados inválidos."""
        invalid_data = {
            'symbol': 'INVALID',
            'price': 'not_a_number'
        }
        
        resp = await self.client.request(
            "POST", "/push",
            json=invalid_data
        )
        
        assert resp.status == 400
```

### Testes de Performance

#### Teste de Carga

```python
# tests/performance/load_test.py
from locust import HttpUser, task, between
import json
import time
import random

class NexusLoadTest(HttpUser):
    wait_time = between(0.1, 0.5)  # 100-500ms entre requests
    
    def on_start(self):
        """Configuração inicial do usuário."""
        self.symbols = ['AAPL-OTC', 'EURUSD-OTC', 'GBPUSD-OTC']
    
    @task(10)
    def send_tick(self):
        """Envia tick para o servidor."""
        symbol = random.choice(self.symbols)
        tick_data = {
            'symbol': symbol,
            'price': round(random.uniform(90, 110), 4),
            'ts': time.time(),
            'volume': random.randint(1, 100),
            'bid_price': round(random.uniform(89, 109), 4),
            'ask_price': round(random.uniform(91, 111), 4)
        }
        
        with self.client.post("/push", json=tick_data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def health_check(self):
        """Verifica health do servidor."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure("Server not healthy")
            else:
                response.failure(f"Status: {response.status_code}")

# Executar teste de carga
# locust -f tests/performance/load_test.py --host=http://localhost:9000
```

#### Teste de Stress

```python
# tests/performance/stress_test.py
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

async def stress_test_server():
    """Teste de stress do servidor."""
    
    async def send_tick(session, symbol, tick_id):
        """Envia um tick individual."""
        tick_data = {
            'symbol': symbol,
            'price': 100.0 + (tick_id % 100) * 0.01,
            'ts': time.time(),
            'volume': 1
        }
        
        start_time = time.time()
        try:
            async with session.post('http://localhost:9000/push', json=tick_data) as resp:
                await resp.json()
                return time.time() - start_time
        except Exception as e:
            print(f"Erro: {e}")
            return None
    
    # Configuração do teste
    concurrent_users = 100
    ticks_per_user = 1000
    symbols = ['AAPL-OTC', 'EURUSD-OTC', 'GBPUSD-OTC']
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Criar tasks para usuários concorrentes
        for user_id in range(concurrent_users):
            for tick_id in range(ticks_per_user):
                symbol = symbols[tick_id % len(symbols)]
                task = send_tick(session, symbol, tick_id)
                tasks.append(task)
        
        print(f"Iniciando teste de stress: {len(tasks)} requests")
        start_time = time.time()
        
        # Executar todas as tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Analisar resultados
        successful_requests = [r for r in results if isinstance(r, float)]
        failed_requests = len(results) - len(successful_requests)
        
        if successful_requests:
            avg_latency = statistics.mean(successful_requests)
            p95_latency = statistics.quantiles(successful_requests, n=20)[18]  # P95
            p99_latency = statistics.quantiles(successful_requests, n=100)[98]  # P99
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        total_time = end_time - start_time
        throughput = len(successful_requests) / total_time
        
        print(f"""
        Resultados do Teste de Stress:
        ==============================
        Total de requests: {len(results)}
        Requests bem-sucedidos: {len(successful_requests)}
        Requests falharam: {failed_requests}
        Taxa de sucesso: {len(successful_requests)/len(results)*100:.2f}%
        
        Latência média: {avg_latency*1000:.2f}ms
        Latência P95: {p95_latency*1000:.2f}ms
        Latência P99: {p99_latency*1000:.2f}ms
        
        Throughput: {throughput:.2f} requests/segundo
        Tempo total: {total_time:.2f} segundos
        """)

# Executar: python tests/performance/stress_test.py
if __name__ == '__main__':
    asyncio.run(stress_test_server())
```

## 🌐 Testes da Extensão (extension/)

### Teste de Interceptação WebSocket

```python
# tests/integration/test_extension.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time

class TestExtension:
    
    @pytest.fixture
    def browser_with_extension(self):
        """Configura navegador com extensão carregada."""
        options = webdriver.ChromeOptions()
        options.add_argument(f"--load-extension=./extension")
        options.add_argument("--disable-web-security")
        
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()
    
    def test_extension_loads(self, browser_with_extension):
        """Testa se a extensão carrega corretamente."""
        driver = browser_with_extension
        
        # Navegar para página de teste
        driver.get("chrome://extensions/")
        
        # Verificar se extensão está listada
        extensions = driver.find_elements(By.CSS_SELECTOR, "[id^='extension-']")
        assert len(extensions) > 0
    
    def test_websocket_interception(self, browser_with_extension):
        """Testa interceptação de WebSocket."""
        driver = browser_with_extension
        
        # Criar página de teste com WebSocket
        test_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test WebSocket</title></head>
        <body>
            <script>
                // Simular WebSocket da corretora
                const ws = new WebSocket('wss://ws-us2.pusher.com:443/app/test');
                
                ws.onopen = function() {
                    console.log('WebSocket conectado');
                    
                    // Simular mensagem de tick
                    setTimeout(() => {
                        ws.send(JSON.stringify({
                            event: 'price_update',
                            data: {
                                symbol: 'AAPL-OTC',
                                price: 100.50,
                                timestamp: Date.now()
                            }
                        }));
                    }, 1000);
                };
                
                // Verificar se extensão interceptou
                window.addEventListener('nexus:tick', (e) => {
                    window.nexusTickReceived = true;
                    window.nexusTickData = e.detail;
                });
            </script>
        </body>
        </html>
        """
        
        # Salvar e carregar página de teste
        with open('/tmp/test_websocket.html', 'w') as f:
            f.write(test_html)
        
        driver.get('file:///tmp/test_websocket.html')
        
        # Aguardar interceptação
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return window.nexusTickReceived === true")
        )
        
        # Verificar dados interceptados
        tick_data = driver.execute_script("return window.nexusTickData")
        assert tick_data is not None
        assert tick_data['symbol'] == 'AAPL-OTC'
        assert tick_data['price'] == 100.50
    
    def test_data_forwarding(self, browser_with_extension):
        """Testa envio de dados para servidor."""
        driver = browser_with_extension
        
        # Configurar mock server para receber dados
        import threading
        import http.server
        import socketserver
        
        received_data = []
        
        class MockHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == '/push':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    received_data.append(json.loads(post_data))
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "received"}')
        
        # Iniciar mock server
        with socketserver.TCPServer(("", 9001), MockHandler) as httpd:
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            # Simular captura de tick
            driver.execute_script("""
                // Simular evento de tick capturado
                window.dispatchEvent(new CustomEvent('nexus:tick', {
                    detail: {
                        symbol: 'AAPL-OTC',
                        price: 100.50,
                        ts: Date.now() / 1000,
                        volume: 1
                    }
                }));
            """)
            
            # Aguardar recebimento no mock server
            time.sleep(2)
            
            # Verificar se dados foram recebidos
            assert len(received_data) > 0
            assert received_data[0]['symbol'] == 'AAPL-OTC'
            
            httpd.shutdown()
```

### Teste de Compatibilidade

```python
def test_browser_compatibility():
    """Testa compatibilidade com diferentes navegadores."""
    
    browsers = [
        ('chrome', webdriver.Chrome),
        ('edge', webdriver.Edge),
        # ('firefox', webdriver.Firefox)  # Se suportado
    ]
    
    for browser_name, browser_class in browsers:
        try:
            options = browser_class.options() if hasattr(browser_class, 'options') else None
            if options:
                options.add_argument(f"--load-extension=./extension")
            
            driver = browser_class(options=options)
            
            # Teste básico de carregamento
            driver.get("chrome://extensions/")
            time.sleep(2)
            
            # Verificar se extensão está ativa
            # Implementar verificação específica por navegador
            
            driver.quit()
            print(f"✅ {browser_name}: Compatível")
            
        except Exception as e:
            print(f"❌ {browser_name}: Erro - {e}")
```

## 🧠 Testes do Modelo IA (model/)

### Testes Unitários do Modelo

```python
# tests/unit/test_model.py
import pytest
import torch
import numpy as np
from model.model import NexusTransformer, ModelConfig, FeatureExtractor

class TestNexusModel:
    
    @pytest.fixture
    def model_config(self):
        """Configuração de teste para o modelo."""
        return ModelConfig(
            d_model=64,
            nhead=4,
            num_layers=2,
            sequence_length=30,
            feature_dim=12
        )
    
    @pytest.fixture
    def model(self, model_config):
        """Instância do modelo para testes."""
        return NexusTransformer(model_config)
    
    def test_model_forward(self, model, model_config):
        """Testa forward pass do modelo."""
        batch_size = 2
        seq_len = model_config.sequence_length
        feature_dim = model_config.feature_dim
        
        # Criar input de teste
        x = torch.randn(batch_size, seq_len, feature_dim)
        
        # Forward pass
        output = model(x)
        
        # Verificar dimensões
        assert output.shape == (batch_size, 2)  # [P(down), P(up)]
        
        # Verificar se output é válido
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_model_probabilities(self, model, model_config):
        """Testa geração de probabilidades."""
        batch_size = 1
        seq_len = model_config.sequence_length
        feature_dim = model_config.feature_dim
        
        x = torch.randn(batch_size, seq_len, feature_dim)
        
        # Obter probabilidades
        probs = model.predict_probabilities(x)
        
        # Verificar se são probabilidades válidas
        assert probs.shape == (batch_size, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_feature_extractor(self):
        """Testa extrator de features."""
        extractor = FeatureExtractor()
        
        # Dados de teste
        tick_data = [
            {'price': 100.0, 'ts': 1234567890, 'volume': 1, 'bid_price': 99.95, 'ask_price': 100.05},
            {'price': 100.1, 'ts': 1234567891, 'volume': 2, 'bid_price': 100.05, 'ask_price': 100.15},
            {'price': 99.9, 'ts': 1234567892, 'volume': 1, 'bid_price': 99.85, 'ask_price': 99.95},
        ]
        
        # Extrair features
        features = extractor.extract_features(tick_data)
        
        # Verificar dimensões
        assert features.shape == (len(tick_data), 12)
        
        # Verificar se features são válidas
        assert not np.isnan(features).any()
        assert np.isfinite(features).all()
    
    def test_model_training_step(self, model, model_config):
        """Testa um passo de treinamento."""
        batch_size = 4
        seq_len = model_config.sequence_length
        feature_dim = model_config.feature_dim
        
        # Dados de entrada
        x = torch.randn(batch_size, seq_len, feature_dim)
        y = torch.randint(0, 2, (batch_size,))  # Labels binários
        
        # Forward pass
        logits = model(x)
        
        # Calcular loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        
        # Verificar se loss é válido
        assert not torch.isnan(loss)
        assert loss.item() > 0
        
        # Backward pass
        loss.backward()
        
        # Verificar se gradientes foram calculados
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
```

### Teste de Backtesting

```python
# tests/integration/test_backtesting.py
import pytest
import pandas as pd
import numpy as np
from model.model import create_nexus_model
from model.infer import create_inference_engine

class TestBacktesting:
    
    @pytest.fixture
    def historical_data(self):
        """Dados históricos para backtesting."""
        # Gerar dados sintéticos para teste
        np.random.seed(42)
        
        dates = pd.date_range('2024-01-01', periods=10000, freq='1min')
        prices = 100 + np.cumsum(np.random.randn(10000) * 0.01)
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            data.append({
                'timestamp': date.timestamp(),
                'price': price,
                'volume': np.random.randint(1, 100),
                'bid_price': price - 0.01,
                'ask_price': price + 0.01
            })
        
        return data
    
    def test_backtesting_accuracy(self, historical_data):
        """Testa precisão do modelo em dados históricos."""
        # Criar modelo
        model = create_nexus_model()
        
        # Simular predições
        correct_predictions = 0
        total_predictions = 0
        
        # Usar janela deslizante para predições
        window_size = 60
        
        for i in range(window_size, len(historical_data) - 20):
            # Sequência de entrada
            sequence = historical_data[i-window_size:i]
            
            # Fazer predição
            prediction = model.predict(sequence)
            
            if prediction['signal']:
                total_predictions += 1
                
                # Verificar resultado real após 20 períodos
                current_price = historical_data[i]['price']
                future_price = historical_data[i + 20]['price']
                
                actual_direction = 'UP' if future_price > current_price else 'DOWN'
                
                if prediction['signal'] == actual_direction:
                    correct_predictions += 1
        
        # Calcular precisão
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print(f"Precisão do backtesting: {accuracy:.2%}")
            print(f"Total de predições: {total_predictions}")
            
            # Verificar se atende critério mínimo
            assert accuracy >= 0.60  # 60% mínimo para dados sintéticos
        else:
            pytest.skip("Nenhuma predição gerada durante o teste")
    
    def test_performance_metrics(self, historical_data):
        """Testa métricas de performance do modelo."""
        engine = create_inference_engine()
        
        # Simular operação por período
        start_time = time.time()
        predictions_made = 0
        
        for i, tick in enumerate(historical_data[:1000]):
            result = asyncio.run(engine.process_tick('TEST-SYMBOL', tick))
            if result and result['prediction']['signal']:
                predictions_made += 1
        
        end_time = time.time()
        
        # Calcular métricas
        total_time = end_time - start_time
        throughput = len(historical_data[:1000]) / total_time
        
        print(f"Throughput: {throughput:.2f} ticks/segundo")
        print(f"Predições geradas: {predictions_made}")
        
        # Verificar performance
        assert throughput >= 50  # Mínimo 50 ticks/segundo
        
        # Verificar estatísticas do engine
        stats = engine.get_performance_stats()
        assert stats['avg_latency_ms'] < 100  # Latência < 100ms
```

## 🌐 Testes da Interface Web (web/)

### Testes de Interface

```python
# tests/e2e/test_web_interface.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class TestWebInterface:
    
    @pytest.fixture
    def browser(self):
        """Configura navegador para testes."""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Executar sem interface gráfica
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()
    
    def test_chart_loads(self, browser):
        """Testa se o gráfico carrega corretamente."""
        driver = browser
        driver.get("http://localhost:8000/chart_comparacao.html")
        
        # Aguardar carregamento do gráfico
        chart_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "chart-container"))
        )
        
        assert chart_container is not None
        
        # Verificar se TradingView foi carregado
        tv_chart = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".tv-lightweight-charts"))
        )
        
        assert tv_chart is not None
    
    def test_websocket_connection(self, browser):
        """Testa conexão WebSocket da interface."""
        driver = browser
        driver.get("http://localhost:8000/chart_comparacao.html")
        
        # Aguardar conexão WebSocket
        time.sleep(3)
        
        # Verificar status da conexão via JavaScript
        connection_status = driver.execute_script("""
            return window.nexusWebSocket && window.nexusWebSocket.readyState === WebSocket.OPEN;
        """)
        
        assert connection_status is True
    
    def test_candle_rendering(self, browser):
        """Testa renderização de candles."""
        driver = browser
        driver.get("http://localhost:8000/chart_comparacao.html")
        
        # Simular recebimento de candle via WebSocket
        driver.execute_script("""
            // Simular candle de teste
            const testCandle = {
                type: 'candle:update',
                symbol: 'AAPL-OTC',
                timeframe: '1m',
                start: Date.now() / 1000,
                open: 100.0,
                high: 101.0,
                low: 99.5,
                close: 100.5,
                volume: 1000
            };
            
            // Simular recebimento via WebSocket
            if (window.handleWebSocketMessage) {
                window.handleWebSocketMessage(testCandle);
            }
        """)
        
        # Aguardar renderização
        time.sleep(2)
        
        # Verificar se candle foi adicionado ao gráfico
        candle_count = driver.execute_script("""
            return window.chartData ? window.chartData.length : 0;
        """)
        
        assert candle_count > 0
    
    def test_responsive_design(self, browser):
        """Testa design responsivo."""
        driver = browser
        
        # Testar diferentes resoluções
        resolutions = [
            (1920, 1080),  # Desktop
            (1366, 768),   # Laptop
            (768, 1024),   # Tablet
            (375, 667)     # Mobile
        ]
        
        for width, height in resolutions:
            driver.set_window_size(width, height)
            driver.get("http://localhost:8000/chart_comparacao.html")
            
            # Aguardar carregamento
            time.sleep(2)
            
            # Verificar se elementos estão visíveis
            chart_container = driver.find_element(By.ID, "chart-container")
            assert chart_container.is_displayed()
            
            # Verificar se gráfico se ajustou ao tamanho
            chart_width = driver.execute_script("""
                return document.getElementById('chart-container').offsetWidth;
            """)
            
            # Gráfico deve ocupar pelo menos 80% da largura
            assert chart_width >= width * 0.8
```

### Testes de Performance Web

```python
# tests/performance/test_web_performance.py
import pytest
import json
import subprocess
from selenium import webdriver

class TestWebPerformance:
    
    def test_lighthouse_performance(self):
        """Testa performance com Google Lighthouse."""
        
        # Executar Lighthouse
        result = subprocess.run([
            'lighthouse',
            'http://localhost:8000/chart_comparacao.html',
            '--output=json',
            '--quiet',
            '--chrome-flags="--headless"'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lighthouse_data = json.loads(result.stdout)
            
            # Verificar métricas de performance
            performance_score = lighthouse_data['lhr']['categories']['performance']['score']
            
            # Métricas específicas
            metrics = lighthouse_data['lhr']['audits']
            
            fcp = metrics['first-contentful-paint']['numericValue']  # First Contentful Paint
            lcp = metrics['largest-contentful-paint']['numericValue']  # Largest Contentful Paint
            cls = metrics['cumulative-layout-shift']['numericValue']  # Cumulative Layout Shift
            
            print(f"Performance Score: {performance_score * 100:.1f}/100")
            print(f"First Contentful Paint: {fcp:.0f}ms")
            print(f"Largest Contentful Paint: {lcp:.0f}ms")
            print(f"Cumulative Layout Shift: {cls:.3f}")
            
            # Critérios de aprovação
            assert performance_score >= 0.8  # Score mínimo 80/100
            assert fcp <= 2000  # FCP <= 2 segundos
            assert lcp <= 4000  # LCP <= 4 segundos
            assert cls <= 0.1   # CLS <= 0.1
        else:
            pytest.skip("Lighthouse não disponível")
    
    def test_memory_usage(self):
        """Testa uso de memória da interface."""
        options = webdriver.ChromeOptions()
        options.add_argument("--enable-memory-info")
        
        driver = webdriver.Chrome(options=options)
        
        try:
            driver.get("http://localhost:8000/chart_comparacao.html")
            
            # Aguardar carregamento completo
            time.sleep(5)
            
            # Simular uso intensivo (muitos candles)
            for i in range(1000):
                driver.execute_script(f"""
                    const candle = {{
                        time: {1234567890 + i * 60},
                        open: {100 + i * 0.01},
                        high: {100.5 + i * 0.01},
                        low: {99.5 + i * 0.01},
                        close: {100.2 + i * 0.01}
                    }};
                    
                    if (window.chart && window.chart.addCandle) {{
                        window.chart.addCandle(candle);
                    }}
                """)
                
                if i % 100 == 0:
                    time.sleep(0.1)  # Pequena pausa
            
            # Verificar uso de memória
            memory_info = driver.execute_script("""
                return {
                    usedJSHeapSize: performance.memory.usedJSHeapSize,
                    totalJSHeapSize: performance.memory.totalJSHeapSize,
                    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                };
            """)
            
            used_mb = memory_info['usedJSHeapSize'] / (1024 * 1024)
            total_mb = memory_info['totalJSHeapSize'] / (1024 * 1024)
            
            print(f"Memória usada: {used_mb:.1f}MB")
            print(f"Memória total: {total_mb:.1f}MB")
            
            # Verificar se uso de memória está dentro do limite
            assert used_mb <= 100  # Máximo 100MB
            
        finally:
            driver.quit()
```

## 🔧 Testes de Ferramentas (tools/)

### Teste do Comparador de Candles

```python
# tests/unit/test_tools.py
import pytest
import json
import tempfile
from tools.compare_candles import CandleComparator, CandleData

class TestCandleComparator:
    
    @pytest.fixture
    def sample_candles_a(self):
        """Candles da fonte A."""
        return [
            {
                'type': 'candle:closed',
                'symbol': 'AAPL-OTC',
                'timeframe': '1m',
                'start': 1234567800,
                'open': 100.0,
                'high': 101.0,
                'low': 99.5,
                'close': 100.5,
                'volume': 1000
            },
            {
                'type': 'candle:closed',
                'symbol': 'AAPL-OTC',
                'timeframe': '1m',
                'start': 1234567860,
                'open': 100.5,
                'high': 101.5,
                'low': 100.0,
                'close': 101.0,
                'volume': 1200
            }
        ]
    
    @pytest.fixture
    def sample_candles_b(self):
        """Candles da fonte B (com pequenas diferenças)."""
        return [
            {
                'type': 'candle:closed',
                'symbol': 'AAPL-OTC',
                'timeframe': '1m',
                'start': 1234567800,
                'open': 100.01,  # Pequena diferença
                'high': 101.0,
                'low': 99.5,
                'close': 100.5,
                'volume': 1000
            },
            {
                'type': 'candle:closed',
                'symbol': 'AAPL-OTC',
                'timeframe': '1m',
                'start': 1234567860,
                'open': 100.5,
                'high': 101.5,
                'low': 100.0,
                'close': 101.0,
                'volume': 1200
            }
        ]
    
    def test_load_candles_from_file(self, sample_candles_a):
        """Testa carregamento de candles de arquivo."""
        comparator = CandleComparator()
        
        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_candles_a, f)
            temp_file = f.name
        
        # Carregar candles
        comparator.load_candles_from_file(temp_file, 'source_a')
        
        # Verificar se foram carregados
        assert 'source_a' in comparator.candles_by_source
        assert 'AAPL-OTC' in comparator.candles_by_source['source_a']
        assert '1m' in comparator.candles_by_source['source_a']['AAPL-OTC']
        assert len(comparator.candles_by_source['source_a']['AAPL-OTC']['1m']) == 2
    
    def test_compare_sources(self, sample_candles_a, sample_candles_b):
        """Testa comparação entre fontes."""
        comparator = CandleComparator(tolerance_pct=0.1)  # 0.1% tolerância
        
        # Carregar dados de ambas as fontes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_candles_a, f)
            file_a = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_candles_b, f)
            file_b = f.name
        
        comparator.load_candles_from_file(file_a, 'source_a')
        comparator.load_candles_from_file(file_b, 'source_b')
        
        # Comparar fontes
        results = comparator.compare_sources('source_a', 'source_b')
        
        # Verificar resultados
        assert len(results) == 2  # Dois candles comparados
        
        # Primeiro candle tem diferença no open
        first_result = results[0]
        assert first_result.open_diff == 0.01
        assert first_result.open_diff_pct > 0
        assert not first_result.is_significant  # Dentro da tolerância
        
        # Segundo candle é idêntico
        second_result = results[1]
        assert second_result.open_diff == 0.0
        assert second_result.high_diff == 0.0
        assert second_result.low_diff == 0.0
        assert second_result.close_diff == 0.0
    
    def test_generate_report(self, sample_candles_a, sample_candles_b):
        """Testa geração de relatório."""
        comparator = CandleComparator()
        
        # Carregar e comparar dados
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_candles_a, f)
            file_a = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_candles_b, f)
            file_b = f.name
        
        comparator.load_candles_from_file(file_a, 'source_a')
        comparator.load_candles_from_file(file_b, 'source_b')
        comparator.compare_sources('source_a', 'source_b')
        
        # Gerar relatório
        report = comparator.generate_report()
        
        # Verificar estrutura do relatório
        assert 'summary' in report
        assert 'statistics' in report
        assert 'worst_cases' in report
        assert 'by_symbol' in report
        assert 'by_timeframe' in report
        
        # Verificar dados do summary
        summary = report['summary']
        assert summary['total_comparisons'] == 2
        assert summary['accuracy_rate_pct'] >= 0
```

### Teste do Sanitizador de Logs

```python
def test_log_sanitizer():
    """Testa sanitização de logs."""
    from tools.sanitize_logs import LogSanitizer
    
    sanitizer = LogSanitizer()
    
    # Texto com dados sensíveis
    sensitive_text = """
    {
        "token": "abc123def456",
        "password": "mypassword123",
        "email": "user@example.com",
        "api_key": "sk-1234567890abcdef",
        "phone": "555-123-4567",
        "url": "https://api.example.com/v1/data"
    }
    """
    
    # Sanitizar
    sanitized = sanitizer._sanitize_text(sensitive_text)
    
    # Verificar se dados sensíveis foram mascarados
    assert "abc123def456" not in sanitized
    assert "mypassword123" not in sanitized
    assert "user@example.com" not in sanitized
    assert "sk-1234567890abcdef" not in sanitized
    assert "555-123-4567" not in sanitized
    
    # Verificar se estrutura foi preservada
    assert "token" in sanitized
    assert "password" in sanitized
    assert "email" in sanitized
```

## 🔄 Testes End-to-End

### Teste do Fluxo Completo

```python
# tests/e2e/test_full_flow.py
import pytest
import asyncio
import time
import json
from selenium import webdriver
import aiohttp

class TestFullFlow:
    
    @pytest.mark.asyncio
    async def test_complete_trading_flow(self):
        """Testa fluxo completo: captura → processamento → predição → sinal."""
        
        # 1. Iniciar servidor
        # (Assumindo que servidor já está rodando)
        
        # 2. Configurar navegador com extensão
        options = webdriver.ChromeOptions()
        options.add_argument("--load-extension=./extension")
        driver = webdriver.Chrome(options=options)
        
        try:
            # 3. Simular captura de ticks
            async with aiohttp.ClientSession() as session:
                # Enviar sequência de ticks
                for i in range(100):
                    tick_data = {
                        'symbol': 'AAPL-OTC',
                        'price': 100.0 + (i % 10) * 0.01,
                        'ts': time.time(),
                        'volume': 1,
                        'bid_price': 99.95 + (i % 10) * 0.01,
                        'ask_price': 100.05 + (i % 10) * 0.01
                    }
                    
                    async with session.post(
                        'http://localhost:9000/push',
                        json=tick_data
                    ) as resp:
                        assert resp.status == 200
                    
                    await asyncio.sleep(0.1)  # 100ms entre ticks
                
                # 4. Verificar se candles foram gerados
                async with session.get('http://localhost:9000/health') as resp:
                    health_data = await resp.json()
                    assert health_data['status'] == 'healthy'
                
                # 5. Conectar WebSocket para receber sinais
                import websockets
                
                signals_received = []
                
                async with websockets.connect('ws://localhost:9000/ws') as websocket:
                    # Aguardar sinais por 30 segundos
                    try:
                        async with asyncio.timeout(30):
                            while True:
                                message = await websocket.recv()
                                data = json.loads(message)
                                
                                if data.get('type') == 'signal':
                                    signals_received.append(data)
                                    print(f"Sinal recebido: {data}")
                                    
                                    # Parar após primeiro sinal
                                    break
                    except asyncio.TimeoutError:
                        pass
                
                # 6. Verificar se pelo menos um sinal foi gerado
                # (Pode não gerar sinal se probabilidade < 80%)
                print(f"Sinais recebidos: {len(signals_received)}")
                
                # 7. Verificar interface web
                driver.get("http://localhost:8000/chart_comparacao.html")
                time.sleep(5)  # Aguardar carregamento
                
                # Verificar se gráfico está funcionando
                chart_loaded = driver.execute_script("""
                    return window.chart !== undefined && window.chartData.length > 0;
                """)
                
                assert chart_loaded, "Gráfico não carregou corretamente"
                
        finally:
            driver.quit()
    
    def test_error_recovery(self):
        """Testa recuperação de erros."""
        # Simular diferentes cenários de erro
        
        # 1. Servidor indisponível
        # 2. Dados corrompidos
        # 3. Falha de rede
        # 4. Extensão desabilitada
        
        # Implementar testes de recuperação
        pass
```

## 📊 Relatórios de Teste

### Geração de Relatórios

```python
# tests/generate_report.py
import pytest
import json
import datetime
from pathlib import Path

def generate_test_report():
    """Gera relatório consolidado de testes."""
    
    # Executar todos os testes com coverage
    result = pytest.main([
        '--cov=server',
        '--cov=model', 
        '--cov=tools',
        '--cov-report=json',
        '--cov-report=html',
        '--json-report',
        '--json-report-file=test_report.json',
        'tests/'
    ])
    
    # Carregar resultados
    with open('test_report.json') as f:
        test_data = json.load(f)
    
    with open('coverage.json') as f:
        coverage_data = json.load(f)
    
    # Gerar relatório consolidado
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'summary': {
            'total_tests': test_data['summary']['total'],
            'passed': test_data['summary']['passed'],
            'failed': test_data['summary']['failed'],
            'skipped': test_data['summary']['skipped'],
            'success_rate': test_data['summary']['passed'] / test_data['summary']['total'] * 100
        },
        'coverage': {
            'total_coverage': coverage_data['totals']['percent_covered'],
            'lines_covered': coverage_data['totals']['covered_lines'],
            'lines_missing': coverage_data['totals']['missing_lines']
        },
        'performance': {
            'total_duration': test_data['duration'],
            'slowest_tests': sorted(
                [(test['nodeid'], test['duration']) for test in test_data['tests']],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    }
    
    # Salvar relatório
    with open('consolidated_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Relatório gerado: consolidated_test_report.json")
    print(f"Taxa de sucesso: {report['summary']['success_rate']:.1f}%")
    print(f"Cobertura de código: {report['coverage']['total_coverage']:.1f}%")

if __name__ == '__main__':
    generate_test_report()
```

## 🚀 Automação de Testes

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Nexus AI Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:alpine
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r server/requirements.txt
        pip install -r tests/requirements.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=server --cov=model
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --timeout=300
    
    - name: Generate coverage report
      run: |
        coverage xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Scripts de Teste Local

```bash
#!/bin/bash
# scripts/run_tests.sh

echo "🧪 Executando testes do Nexus AI..."

# Verificar se servidor está rodando
if ! curl -s http://localhost:9000/health > /dev/null; then
    echo "❌ Servidor não está rodando. Iniciando..."
    cd server && python server.py &
    SERVER_PID=$!
    sleep 5
else
    echo "✅ Servidor já está rodando"
fi

# Verificar se interface web está disponível
if ! curl -s http://localhost:8000 > /dev/null; then
    echo "❌ Interface web não está rodando. Iniciando..."
    cd web && python -m http.server 8000 &
    WEB_PID=$!
    sleep 3
else
    echo "✅ Interface web já está rodando"
fi

# Executar testes
echo "🔧 Executando testes unitários..."
pytest tests/unit/ -v

echo "🔗 Executando testes de integração..."
pytest tests/integration/ -v

echo "⚡ Executando testes de performance..."
pytest tests/performance/ -v --timeout=300

echo "🌐 Executando testes E2E..."
pytest tests/e2e/ -v --timeout=600

# Gerar relatório
echo "📊 Gerando relatório..."
python tests/generate_report.py

# Cleanup
if [ ! -z "$SERVER_PID" ]; then
    kill $SERVER_PID
fi

if [ ! -z "$WEB_PID" ]; then
    kill $WEB_PID
fi

echo "✅ Testes concluídos!"
```

## 📋 Checklist de Validação

### Antes do Deploy

- [ ] Todos os testes unitários passando (100%)
- [ ] Testes de integração passando (>95%)
- [ ] Testes E2E passando (>90%)
- [ ] Cobertura de código >80%
- [ ] Performance dentro dos SLAs
- [ ] Segurança validada
- [ ] Documentação atualizada

### Critérios de Aceitação

- [ ] Latência E2E <100ms
- [ ] Precisão do modelo >80%
- [ ] Uptime >99.9%
- [ ] Throughput >100 ticks/segundo
- [ ] Interface responsiva
- [ ] Extensão compatível com Chrome/Edge
- [ ] Logs sanitizados automaticamente

---

Este guia de testes fornece uma base sólida para validar todos os aspectos do sistema Nexus AI, garantindo qualidade, performance e confiabilidade em produção.

