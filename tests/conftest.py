"""
Configuração base para testes do Nexus AI.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path

# Configurar event loop para testes assíncronos
@pytest.fixture(scope="session")
def event_loop():
    """Cria event loop para testes assíncronos."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_ticks():
    """Dados de ticks para testes."""
    return [
        {
            'symbol': 'AAPL-OTC',
            'price': 100.0,
            'ts': 1234567890,
            'volume': 1,
            'bid_price': 99.95,
            'ask_price': 100.05
        },
        {
            'symbol': 'AAPL-OTC', 
            'price': 100.1,
            'ts': 1234567891,
            'volume': 2,
            'bid_price': 100.05,
            'ask_price': 100.15
        },
        {
            'symbol': 'AAPL-OTC',
            'price': 99.9,
            'ts': 1234567892,
            'volume': 1,
            'bid_price': 99.85,
            'ask_price': 99.95
        }
    ]

@pytest.fixture
def sample_candles():
    """Dados de candles para testes."""
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
def temp_json_file():
    """Cria arquivo JSON temporário."""
    def _create_temp_file(data):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name
    return _create_temp_file

@pytest.fixture
def mock_server_config():
    """Configuração para servidor de teste."""
    return {
        'host': 'localhost',
        'port': 9001,
        'debug': True
    }

# Configurações globais para testes
pytest_plugins = ['pytest_asyncio']

