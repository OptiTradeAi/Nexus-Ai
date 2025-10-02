"""
Nexus AI - Módulo de Inferência em Tempo Real
Sistema de inferência otimizado para baixa latência em produção.

Funcionalidades:
- Inferência em tempo real com cache
- Batch processing para múltiplos símbolos
- Monitoramento de performance
- Integração com servidor principal

Autor: Manus AI
Data: 2025-09-29
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import threading
from dataclasses import dataclass
import json

import torch
import numpy as np

from .model import NexusAI, ModelConfig, create_nexus_model

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuração do sistema de inferência."""
    # Performance
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    max_sequence_cache: int = 1000
    
    # Modelo
    model_path: Optional[str] = None
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    # Monitoramento
    log_predictions: bool = True
    performance_window: int = 100

class PerformanceMonitor:
    """Monitor de performance para inferência."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions_count = 0
        self.errors_count = 0
        self.start_time = time.time()
    
    def record_prediction(self, latency_ms: float, success: bool = True):
        """Registra uma predição."""
        self.latencies.append(latency_ms)
        self.predictions_count += 1
        if not success:
            self.errors_count += 1
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de performance."""
        if not self.latencies:
            return {
                'avg_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
                'predictions_per_sec': 0,
                'error_rate': 0,
                'uptime_seconds': time.time() - self.start_time
            }
        
        latencies = list(self.latencies)
        uptime = time.time() - self.start_time
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'predictions_per_sec': self.predictions_count / max(uptime, 1),
            'error_rate': self.errors_count / max(self.predictions_count, 1),
            'uptime_seconds': uptime
        }

class SequenceCache:
    """Cache otimizado para sequências de ticks."""
    
    def __init__(self, max_size: int = 1000, sequence_length: int = 60):
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.cache = defaultdict(lambda: deque(maxlen=sequence_length))
        self.last_access = {}
    
    def add_tick(self, symbol: str, tick: Dict):
        """Adiciona tick à sequência do símbolo."""
        self.cache[symbol].append(tick)
        self.last_access[symbol] = time.time()
        
        # Cleanup de símbolos antigos
        self._cleanup_old_symbols()
    
    def get_sequence(self, symbol: str) -> List[Dict]:
        """Retorna sequência de ticks para o símbolo."""
        self.last_access[symbol] = time.time()
        return list(self.cache[symbol])
    
    def _cleanup_old_symbols(self):
        """Remove símbolos não acessados recentemente."""
        if len(self.cache) <= self.max_size:
            return
        
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hora
        
        symbols_to_remove = [
            symbol for symbol, last_time in self.last_access.items()
            if last_time < cutoff_time
        ]
        
        for symbol in symbols_to_remove:
            del self.cache[symbol]
            del self.last_access[symbol]

class BatchProcessor:
    """Processador de lotes para inferência eficiente."""
    
    def __init__(self, model: NexusAI, config: InferenceConfig):
        self.model = model
        self.config = config
        self.pending_requests = []
        self.batch_timer = None
        self.lock = threading.Lock()
    
    async def predict_async(self, symbol: str, tick_sequence: List[Dict]) -> Dict:
        """Adiciona predição ao lote e retorna resultado."""
        future = asyncio.Future()
        
        with self.lock:
            self.pending_requests.append({
                'symbol': symbol,
                'sequence': tick_sequence,
                'future': future
            })
            
            # Processar imediatamente se lote estiver cheio
            if len(self.pending_requests) >= self.config.max_batch_size:
                asyncio.create_task(self._process_batch())
            elif self.batch_timer is None:
                # Agendar processamento do lote
                self.batch_timer = asyncio.create_task(
                    self._schedule_batch_processing()
                )
        
        return await future
    
    async def _schedule_batch_processing(self):
        """Agenda processamento do lote após timeout."""
        await asyncio.sleep(self.config.batch_timeout_ms / 1000.0)
        await self._process_batch()
    
    async def _process_batch(self):
        """Processa lote de predições."""
        with self.lock:
            if not self.pending_requests:
                return
            
            batch = self.pending_requests.copy()
            self.pending_requests.clear()
            
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None
        
        # Processar lote em thread separada para não bloquear
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._process_batch_sync, batch)
    
    def _process_batch_sync(self, batch: List[Dict]):
        """Processa lote de forma síncrona."""
        try:
            for request in batch:
                try:
                    # Predição individual (pode ser otimizada para batch real)
                    prediction = self.model.predict(request['sequence'])
                    request['future'].set_result(prediction)
                except Exception as e:
                    request['future'].set_exception(e)
        except Exception as e:
            # Falha geral do lote
            for request in batch:
                if not request['future'].done():
                    request['future'].set_exception(e)

class NexusInferenceEngine:
    """
    Engine principal de inferência do Nexus AI.
    
    Funcionalidades:
    - Inferência em tempo real com baixa latência
    - Cache de sequências por símbolo
    - Batch processing automático
    - Monitoramento de performance
    - Geração de sinais de trading
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config.performance_window)
        self.sequence_cache = SequenceCache(
            max_size=config.max_sequence_cache,
            sequence_length=60  # Será atualizado com config do modelo
        )
        
        # Inicializar modelo
        self._initialize_model()
        
        # Batch processor
        self.batch_processor = BatchProcessor(self.model, config)
        
        # Estado
        self.is_running = False
        self.stats_lock = threading.Lock()
        
        logger.info("Nexus Inference Engine inicializado")
    
    def _initialize_model(self):
        """Inicializa o modelo de IA."""
        try:
            # Configuração do modelo
            model_config_overrides = {}
            
            if self.config.device != 'auto':
                # Device será configurado no modelo
                pass
            
            # Criar modelo
            self.model = create_nexus_model(model_config_overrides)
            
            # Carregar pesos se fornecido
            if self.config.model_path:
                self.model.load_model(self.config.model_path)
            
            # Atualizar cache com sequence_length do modelo
            self.sequence_cache.sequence_length = self.model.config.sequence_length
            
            logger.info(f"Modelo carregado: {self.model.config}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo: {e}")
            raise
    
    async def process_tick(self, symbol: str, tick: Dict) -> Optional[Dict]:
        """
        Processa um tick e retorna predição/sinal se aplicável.
        
        Args:
            symbol: Símbolo do ativo (ex: 'AAPL-OTC')
            tick: Dict com {price, ts, volume?, bid?, ask?}
        
        Returns:
            result: Dict com predição e sinal (se gerado) ou None
        """
        start_time = time.time()
        
        try:
            # Adicionar tick ao cache
            self.sequence_cache.add_tick(symbol, tick)
            
            # Obter sequência para predição
            sequence = self.sequence_cache.get_sequence(symbol)
            
            if len(sequence) < 10:  # Mínimo de ticks para predição
                return None
            
            # Realizar predição (via batch processor)
            prediction = await self.batch_processor.predict_async(symbol, sequence)
            
            # Verificar se deve gerar sinal
            signal = self.model.generate_signal(symbol, prediction)
            
            # Registrar performance
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_prediction(latency_ms, True)
            
            # Log se configurado
            if self.config.log_predictions and signal:
                logger.info(f"Sinal gerado: {signal['signal_id']}")
            
            result = {
                'symbol': symbol,
                'prediction': prediction,
                'signal': signal,
                'latency_ms': latency_ms,
                'sequence_length': len(sequence)
            }
            
            return result
            
        except Exception as e:
            # Registrar erro
            latency_ms = (time.time() - start_time) * 1000
            self.performance_monitor.record_prediction(latency_ms, False)
            
            logger.error(f"Erro ao processar tick {symbol}: {e}")
            return None
    
    async def predict_batch(self, requests: List[Tuple[str, List[Dict]]]) -> List[Dict]:
        """
        Processa lote de predições.
        
        Args:
            requests: Lista de (symbol, tick_sequence)
        
        Returns:
            predictions: Lista de predições
        """
        tasks = []
        
        for symbol, sequence in requests:
            task = self.batch_processor.predict_async(symbol, sequence)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Processar resultados
        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Erro na predição {i}: {result}")
                predictions.append(None)
            else:
                predictions.append(result)
        
        return predictions
    
    def get_performance_stats(self) -> Dict:
        """Retorna estatísticas de performance."""
        with self.stats_lock:
            stats = self.performance_monitor.get_stats()
            
            # Adicionar estatísticas do cache
            stats.update({
                'cached_symbols': len(self.sequence_cache.cache),
                'total_sequences': sum(len(seq) for seq in self.sequence_cache.cache.values()),
                'model_device': str(self.model.device),
                'model_config': {
                    'sequence_length': self.model.config.sequence_length,
                    'probability_threshold': self.model.config.probability_threshold,
                    'throttle_seconds': self.model.config.throttle_seconds
                }
            })
            
            return stats
    
    def get_signal_history(self, limit: int = 50) -> List[Dict]:
        """Retorna histórico de sinais gerados."""
        return self.model.signal_history[-limit:]
    
    def update_model_config(self, config_updates: Dict):
        """Atualiza configuração do modelo em tempo real."""
        try:
            for key, value in config_updates.items():
                if hasattr(self.model.config, key):
                    setattr(self.model.config, key, value)
                    logger.info(f"Configuração atualizada: {key} = {value}")
        except Exception as e:
            logger.error(f"Erro ao atualizar configuração: {e}")
    
    async def shutdown(self):
        """Finaliza o engine de inferência."""
        self.is_running = False
        
        # Cancelar batch timer se ativo
        if self.batch_processor.batch_timer:
            self.batch_processor.batch_timer.cancel()
        
        logger.info("Nexus Inference Engine finalizado")

# Função de conveniência para criar engine
def create_inference_engine(
    model_path: Optional[str] = None,
    config_overrides: Optional[Dict] = None
) -> NexusInferenceEngine:
    """
    Cria engine de inferência com configuração padrão.
    
    Args:
        model_path: Caminho para modelo treinado (opcional)
        config_overrides: Configurações para sobrescrever
    
    Returns:
        engine: Engine de inferência configurado
    """
    config = InferenceConfig()
    
    if model_path:
        config.model_path = model_path
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return NexusInferenceEngine(config)

# Exemplo de uso
async def example_usage():
    """Exemplo de como usar o engine de inferência."""
    
    # Criar engine
    engine = create_inference_engine()
    
    # Simular ticks
    for i in range(100):
        tick = {
            'price': 100.0 + np.random.randn() * 0.1,
            'ts': time.time(),
            'volume': 1,
            'bid_price': 99.95,
            'ask_price': 100.05
        }
        
        result = await engine.process_tick('AAPL-OTC', tick)
        
        if result and result['signal']:
            print(f"Sinal: {result['signal']}")
        
        await asyncio.sleep(0.1)  # 100ms entre ticks
    
    # Estatísticas
    stats = engine.get_performance_stats()
    print(f"Performance: {stats}")
    
    # Finalizar
    await engine.shutdown()

if __name__ == '__main__':
    # Executar exemplo
    asyncio.run(example_usage())

