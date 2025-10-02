"""
Nexus AI - Modelo de Predição de Candles
Sistema de IA para predição de direção de candles em opções binárias.

Arquitetura:
- Transformer temporal para sequências de ticks/candles
- Features: velocidade, imbalance, micro-volatilidade, shape
- Output: probabilidades P(up), P(down)
- Threshold: >= 80% para sinalização

Autor: Manus AI
Data: 2025-09-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuração do modelo Nexus AI."""
    # Arquitetura
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # Dados
    sequence_length: int = 60  # 60 ticks/candles
    feature_dim: int = 12  # Número de features por tick
    
    # Treinamento
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 1e-5
    
    # Predição
    probability_threshold: float = 0.80
    lead_time_seconds: int = 20
    throttle_seconds: int = 300  # 5 minutos

class PositionalEncoding(nn.Module):
    """Codificação posicional para Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class NexusTransformer(nn.Module):
    """
    Modelo Transformer para predição de direção de candles.
    
    Arquitetura:
    1. Embedding das features de entrada
    2. Positional encoding
    3. Stack de Transformer layers
    4. Cabeça de classificação (up/down)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Cabeça de classificação
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 2)  # [P(down), P(up)]
        )
        
        # Inicialização dos pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Inicialização Xavier para melhor convergência."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de shape (batch_size, seq_len, feature_dim)
            mask: Máscara de padding (opcional)
        
        Returns:
            logits: Tensor de shape (batch_size, 2) com logits [P(down), P(up)]
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling
        if mask is not None:
            # Mascarar posições de padding
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = encoded.sum(dim=1) / lengths
        else:
            pooled = encoded.mean(dim=1)
        
        # Classificação
        logits = self.classifier(pooled)
        
        return logits
    
    def predict_probabilities(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predição de probabilidades com softmax.
        
        Returns:
            probs: Tensor de shape (batch_size, 2) com [P(down), P(up)]
        """
        with torch.no_grad():
            logits = self.forward(x, mask)
            probs = F.softmax(logits, dim=-1)
        return probs

class FeatureExtractor:
    """
    Extrator de features para ticks e candles.
    
    Features extraídas:
    1. Preço normalizado
    2. Retornos (1, 5, 10 períodos)
    3. Velocidade de mudança
    4. Imbalance bid/ask
    5. Micro-volatilidade
    6. Shape do candle (se disponível)
    7. Volume relativo
    8. Momentum
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.price_history = []
        self.volume_history = []
    
    def extract_features(self, tick_data: List[Dict]) -> np.ndarray:
        """
        Extrai features de uma sequência de ticks.
        
        Args:
            tick_data: Lista de dicts com {price, ts, volume?, bid?, ask?}
        
        Returns:
            features: Array de shape (len(tick_data), feature_dim)
        """
        if len(tick_data) == 0:
            return np.array([]).reshape(0, 12)
        
        features = []
        
        for i, tick in enumerate(tick_data):
            price = float(tick['price'])
            ts = float(tick['ts'])
            volume = float(tick.get('volume', 1))
            bid = float(tick.get('bid_price', price))
            ask = float(tick.get('ask_price', price))
            
            # Manter histórico
            self.price_history.append(price)
            self.volume_history.append(volume)
            
            # Limitar tamanho do histórico
            if len(self.price_history) > self.lookback_window:
                self.price_history.pop(0)
                self.volume_history.pop(0)
            
            # Calcular features
            feature_vector = self._compute_features(
                price, ts, volume, bid, ask, i, len(tick_data)
            )
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _compute_features(self, price: float, ts: float, volume: float, 
                         bid: float, ask: float, idx: int, total_len: int) -> List[float]:
        """Computa vetor de features para um tick."""
        
        # 1. Preço normalizado (z-score)
        if len(self.price_history) > 1:
            price_mean = np.mean(self.price_history)
            price_std = np.std(self.price_history) + 1e-8
            price_norm = (price - price_mean) / price_std
        else:
            price_norm = 0.0
        
        # 2-4. Retornos (1, 5, 10 períodos)
        returns_1 = self._compute_return(1) if len(self.price_history) > 1 else 0.0
        returns_5 = self._compute_return(5) if len(self.price_history) > 5 else 0.0
        returns_10 = self._compute_return(10) if len(self.price_history) > 10 else 0.0
        
        # 5. Velocidade de mudança (derivada)
        velocity = self._compute_velocity() if len(self.price_history) > 2 else 0.0
        
        # 6. Imbalance bid/ask
        spread = ask - bid
        mid_price = (bid + ask) / 2
        imbalance = (price - mid_price) / (spread + 1e-8) if spread > 0 else 0.0
        
        # 7. Micro-volatilidade (rolling std)
        micro_vol = self._compute_micro_volatility() if len(self.price_history) > 5 else 0.0
        
        # 8. Volume relativo
        if len(self.volume_history) > 1:
            vol_mean = np.mean(self.volume_history)
            vol_relative = volume / (vol_mean + 1e-8)
        else:
            vol_relative = 1.0
        
        # 9. Momentum (média móvel de retornos)
        momentum = self._compute_momentum() if len(self.price_history) > 10 else 0.0
        
        # 10. Posição na sequência (normalizada)
        seq_position = idx / max(total_len - 1, 1)
        
        # 11-12. Features temporais (sin/cos do timestamp)
        time_sin = np.sin(2 * np.pi * (ts % 3600) / 3600)  # Ciclo horário
        time_cos = np.cos(2 * np.pi * (ts % 3600) / 3600)
        
        return [
            price_norm, returns_1, returns_5, returns_10, velocity,
            imbalance, micro_vol, vol_relative, momentum, seq_position,
            time_sin, time_cos
        ]
    
    def _compute_return(self, periods: int) -> float:
        """Calcula retorno logarítmico."""
        if len(self.price_history) <= periods:
            return 0.0
        
        current = self.price_history[-1]
        past = self.price_history[-periods-1]
        
        return np.log(current / past) if past > 0 else 0.0
    
    def _compute_velocity(self) -> float:
        """Calcula velocidade de mudança (derivada)."""
        if len(self.price_history) < 3:
            return 0.0
        
        # Derivada simples
        p1, p2, p3 = self.price_history[-3:]
        return (p3 - p1) / 2.0
    
    def _compute_micro_volatility(self) -> float:
        """Calcula micro-volatilidade (rolling std)."""
        if len(self.price_history) < 5:
            return 0.0
        
        recent_prices = self.price_history[-5:]
        returns = [np.log(recent_prices[i] / recent_prices[i-1]) 
                  for i in range(1, len(recent_prices))
                  if recent_prices[i-1] > 0]
        
        return np.std(returns) if returns else 0.0
    
    def _compute_momentum(self) -> float:
        """Calcula momentum (média móvel de retornos)."""
        if len(self.price_history) < 10:
            return 0.0
        
        recent_prices = self.price_history[-10:]
        returns = [np.log(recent_prices[i] / recent_prices[i-1]) 
                  for i in range(1, len(recent_prices))
                  if recent_prices[i-1] > 0]
        
        return np.mean(returns) if returns else 0.0

class NexusAI:
    """
    Classe principal do sistema de IA Nexus.
    
    Funcionalidades:
    - Carregamento/salvamento de modelo
    - Inferência em tempo real
    - Geração de sinais de trading
    - Aplicação de regras de negócio
    """
    
    def __init__(self, config: ModelConfig, model_path: Optional[str] = None):
        self.config = config
        self.model = NexusTransformer(config)
        self.feature_extractor = FeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Estado de sinalização
        self.last_signal_time = {}  # symbol -> timestamp
        self.signal_history = []
        
        # Carregar modelo se fornecido
        if model_path:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Nexus AI inicializado no device: {self.device}")
    
    def load_model(self, model_path: str):
        """Carrega modelo treinado."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Modelo carregado de: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def save_model(self, model_path: str, metadata: Optional[Dict] = None):
        """Salva modelo treinado."""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'metadata': metadata or {}
            }
            torch.save(checkpoint, model_path)
            logger.info(f"Modelo salvo em: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise
    
    def predict(self, tick_sequence: List[Dict]) -> Dict:
        """
        Realiza predição para uma sequência de ticks.
        
        Args:
            tick_sequence: Lista de ticks com {price, ts, volume?, bid?, ask?}
        
        Returns:
            prediction: Dict com {prob_up, prob_down, signal, confidence}
        """
        try:
            # Extrair features
            features = self.feature_extractor.extract_features(tick_sequence)
            
            if len(features) == 0:
                return self._empty_prediction()
            
            # Preparar tensor
            if len(features) < self.config.sequence_length:
                # Padding com zeros
                padding = np.zeros((self.config.sequence_length - len(features), 
                                  self.config.feature_dim))
                features = np.vstack([padding, features])
                mask = torch.zeros(self.config.sequence_length, dtype=torch.bool)
                mask[:self.config.sequence_length - len(tick_sequence)] = True
            else:
                # Truncar para tamanho máximo
                features = features[-self.config.sequence_length:]
                mask = None
            
            # Converter para tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            if mask is not None:
                mask = mask.unsqueeze(0).to(self.device)
            
            # Predição
            probs = self.model.predict_probabilities(x, mask)
            prob_down, prob_up = probs[0].cpu().numpy()
            
            # Determinar sinal
            max_prob = max(prob_up, prob_down)
            signal = None
            
            if max_prob >= self.config.probability_threshold:
                signal = 'UP' if prob_up > prob_down else 'DOWN'
            
            return {
                'prob_up': float(prob_up),
                'prob_down': float(prob_down),
                'signal': signal,
                'confidence': float(max_prob),
                'timestamp': tick_sequence[-1]['ts'] if tick_sequence else None
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return self._empty_prediction()
    
    def should_signal(self, symbol: str, prediction: Dict) -> bool:
        """
        Verifica se deve emitir sinal baseado nas regras de negócio.
        
        Regras:
        1. Probabilidade >= threshold
        2. Throttle entre sinais
        3. Lead time adequado
        """
        if not prediction['signal']:
            return False
        
        current_time = prediction['timestamp']
        if not current_time:
            return False
        
        # Verificar throttle
        last_signal = self.last_signal_time.get(symbol, 0)
        if current_time - last_signal < self.config.throttle_seconds:
            return False
        
        # Verificar confidence
        if prediction['confidence'] < self.config.probability_threshold:
            return False
        
        return True
    
    def generate_signal(self, symbol: str, prediction: Dict) -> Optional[Dict]:
        """
        Gera sinal de trading se as condições forem atendidas.
        
        Returns:
            signal: Dict com informações do sinal ou None
        """
        if not self.should_signal(symbol, prediction):
            return None
        
        current_time = prediction['timestamp']
        expiry_time = current_time + self.config.lead_time_seconds
        
        signal = {
            'symbol': symbol,
            'direction': prediction['signal'],
            'probability': prediction['confidence'],
            'entry_time': current_time,
            'expiry_time': expiry_time,
            'lead_time': self.config.lead_time_seconds,
            'model_version': '1.0.0',
            'signal_id': f"{symbol}_{int(current_time)}_{prediction['signal']}"
        }
        
        # Atualizar estado
        self.last_signal_time[symbol] = current_time
        self.signal_history.append(signal)
        
        # Manter apenas últimos 100 sinais
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
        
        logger.info(f"Sinal gerado: {signal['signal_id']} - {signal['direction']} @ {signal['probability']:.3f}")
        
        return signal
    
    def _empty_prediction(self) -> Dict:
        """Retorna predição vazia."""
        return {
            'prob_up': 0.5,
            'prob_down': 0.5,
            'signal': None,
            'confidence': 0.5,
            'timestamp': None
        }

# Função de conveniência para criar modelo
def create_nexus_model(config_overrides: Optional[Dict] = None) -> NexusAI:
    """
    Cria instância do Nexus AI com configuração padrão.
    
    Args:
        config_overrides: Dict com configurações para sobrescrever
    
    Returns:
        nexus_ai: Instância configurada do Nexus AI
    """
    config = ModelConfig()
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return NexusAI(config)

