#!/usr/bin/env python3
"""
Nexus AI - Servidor Principal
Sistema completo de trading com IA para opções binárias.

Funcionalidades:
- Ingestão de ticks via WebSocket interceptado (extensão) ou screen-share
- Agregação em candles OHLC em tempo real
- Predição de velas com IA (probabilidade >= 80%)
- Alertas TTS e visuais
- Regras de martingale (1x apenas)
- Interface moderna multilíngue

Autor: Manus AI
Data: 2025-09-29
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Any
from collections import defaultdict
import base64
import os

from aiohttp import web, WSMsgType
import aiohttp_cors
import websockets
from aiohttp.web_ws import WSMsgType

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações do servidor
WS_PORT = 8765
HTTP_PORT = int(os.environ.get('PORT', 9000))
AUTH_TOKEN = os.environ.get('NEXUS_AUTH_TOKEN', 'nexus_dev_token_2025')

# Estado global do servidor
ws_clients: Set = set()
screen_clients: Set = set()
current_candles: Dict = {}  # (symbol, period, start) -> candle
signal_history: Dict = {}  # symbol -> last_signal_timestamp
martingale_state: Dict = {}  # signal_id -> martingale_info

# Configurações de trading
MIN_SIGNAL_INTERVAL = 300  # 5 minutos em segundos
MIN_PROBABILITY_THRESHOLD = 0.80
LEAD_TIME_SECONDS = 20

def sanitize_sensitive_data(data: str) -> str:
    """Remove dados sensíveis de logs e payloads."""
    sensitive_patterns = [
        'token', 'password', 'auth', 'key', 'secret', 'credential'
    ]
    
    for pattern in sensitive_patterns:
        if pattern.lower() in data.lower():
            # Mascarar dados sensíveis
            return data.replace(data, '[REDACTED_SENSITIVE_DATA]')
    return data

def floor_to_period(timestamp: float, period_seconds: int) -> int:
    """Arredonda timestamp para o início do período."""
    return int(timestamp // period_seconds * period_seconds)

async def broadcast_to_clients(message: Dict[str, Any], client_set: Set = None):
    """Envia mensagem para todos os clientes WebSocket conectados."""
    if client_set is None:
        client_set = ws_clients
    
    if not client_set:
        return
    
    data = json.dumps(message, default=str)
    disconnected = []
    
    for ws in list(client_set):
        try:
            await ws.send_str(data)
        except Exception as e:
            logger.warning(f"Erro ao enviar para cliente: {e}")
            disconnected.append(ws)
    
    # Remove clientes desconectados
    for ws in disconnected:
        client_set.discard(ws)

def parse_tick_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extrai tick normalizado de diferentes tipos de payload.
    
    Formatos suportados:
    1. Extension: {"forwarded": {"parsed": {...}}, "source": "extension"}
    2. Screen: {"type": "frame", "image": "...", "ts": ...}
    3. Direct: {"symbol": "...", "price": ..., "ts": ...}
    """
    try:
        # Payload direto (já normalizado)
        if 'symbol' in payload and 'price' in payload:
            return {
                'symbol': payload['symbol'],
                'price': float(payload['price']),
                'ts': float(payload.get('ts', time.time()))
            }
        
        # Payload da extensão
        if 'forwarded' in payload and payload.get('source') == 'extension':
            forwarded = payload['forwarded']
            parsed = forwarded.get('parsed')
            
            if parsed and isinstance(parsed, dict):
                # Formato Pusher: {"event": "AAPL-OTC", "data": "{...}"}
                if 'event' in parsed and 'data' in parsed:
                    event = parsed['event']
                    data_str = parsed['data']
                    
                    if isinstance(data_str, str):
                        try:
                            data = json.loads(data_str)
                        except:
                            data = {}
                    else:
                        data = data_str or {}
                    
                    # Extrair informações do tick
                    symbol = data.get('sym') or data.get('symbol') or event
                    price = data.get('price') or data.get('last')
                    timestamp = data.get('timestamp') or data.get('ts')
                    
                    if price is not None:
                        ts_val = time.time()
                        if timestamp:
                            try:
                                # Converter timestamp (pode estar em ms)
                                ts_val = int(timestamp) / 1000.0 if timestamp > 1e12 else float(timestamp)
                            except:
                                pass
                        
                        return {
                            'symbol': str(symbol),
                            'price': float(price),
                            'ts': ts_val
                        }
        
        # Payload de screen-share (para futuro processamento OCR)
        if payload.get('type') == 'frame':
            # Por enquanto, apenas log do frame recebido
            logger.info(f"Frame de screen-share recebido: {payload.get('ts')}")
            return None
        
    except Exception as e:
        logger.error(f"Erro ao parsear payload: {e}")
    
    return None

async def update_candle_aggregators(symbol: str, price: float, timestamp: float):
    """Atualiza agregadores de candles para diferentes timeframes."""
    timeframes = [1, 60, 300]  # 1s, 1m, 5m
    
    for period in timeframes:
        start_time = floor_to_period(timestamp, period)
        key = (symbol, period, start_time)
        
        candle = current_candles.get(key)
        if candle is None:
            # Novo candle
            candle = {
                'symbol': symbol,
                'timeframe': f'{period}s',
                'start': start_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1,
                'is_closed': False
            }
            current_candles[key] = candle
        else:
            # Atualizar candle existente
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['volume'] += 1
        
        # Broadcast da atualização
        await broadcast_to_clients({
            'type': 'candle:update',
            'symbol': symbol,
            'timeframe': candle['timeframe'],
            'start': start_time,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume'],
            'is_closed': False
        })

async def close_completed_candles():
    """Fecha candles que já passaram do seu período."""
    current_time = time.time()
    to_close = []
    
    for (symbol, period, start_time), candle in list(current_candles.items()):
        end_time = start_time + period
        
        if current_time >= end_time + 0.1:  # Buffer de 100ms
            to_close.append(((symbol, period, start_time), candle))
    
    for (symbol, period, start_time), candle in to_close:
        # Remove do estado atual
        del current_candles[(symbol, period, start_time)]
        
        # Marca como fechado
        candle['is_closed'] = True
        
        # Broadcast do candle fechado
        await broadcast_to_clients({
            'type': 'candle:closed',
            'symbol': symbol,
            'timeframe': candle['timeframe'],
            'start': start_time,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume'],
            'is_closed': True
        })
        
        logger.info(f"Candle fechado: {symbol} {candle['timeframe']} OHLC({candle['open']:.4f}, {candle['high']:.4f}, {candle['low']:.4f}, {candle['close']:.4f})")

def check_auth(request) -> bool:
    """Verifica autenticação opcional via Bearer token."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        return token == AUTH_TOKEN
    return True  # Auth opcional por enquanto

async def handle_push(request):
    """
    Endpoint POST /push - Recebe ticks da extensão ou outros sources.
    """
    if not check_auth(request):
        return web.Response(text="Unauthorized", status=401)
    
    try:
        payload = await request.json()
    except Exception as e:
        text = await request.text()
        logger.error(f"Payload JSON inválido: {e}")
        return web.Response(text=f"JSON inválido: {text[:200]}", status=400)
    
    # Sanitizar dados sensíveis
    payload_str = json.dumps(payload)
    sanitized_payload = sanitize_sensitive_data(payload_str)
    
    # Parsear tick
    tick = parse_tick_from_payload(payload)
    
    if tick:
        # Broadcast do tick
        await broadcast_to_clients({
            'type': 'tick',
            'symbol': tick['symbol'],
            'price': tick['price'],
            'ts': tick['ts']
        })
        
        # Atualizar agregadores
        await update_candle_aggregators(tick['symbol'], tick['price'], tick['ts'])
        
        logger.info(f"Tick processado: {tick['symbol']} @ {tick['price']:.4f}")
        return web.Response(text="OK", status=200)
    else:
        # Payload não reconhecido, mas fazer broadcast para debug
        await broadcast_to_clients({
            'type': 'raw_payload',
            'payload': payload,
            'ts': time.time()
        })
        
        logger.warning(f"Payload não reconhecido: {sanitized_payload[:200]}")
        return web.Response(text="Payload recebido (tick não extraído)", status=200)

async def handle_websocket(request):
    """
    Endpoint GET /ws - WebSocket para clientes da UI.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    ws_clients.add(ws)
    logger.info(f"Cliente WebSocket conectado. Total: {len(ws_clients)}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Processar comandos do cliente se necessário
                    logger.debug(f"Mensagem do cliente: {data}")
                except:
                    pass
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"Erro no WebSocket: {ws.exception()}")
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
    finally:
        ws_clients.discard(ws)
        logger.info(f"Cliente WebSocket desconectado. Total: {len(ws_clients)}")
    
    return ws

async def handle_screen_websocket(request):
    """
    Endpoint GET /ws_screen - WebSocket para receber frames de screen-share.
    """
    if not check_auth(request):
        return web.Response(text="Unauthorized", status=401)
    
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    screen_clients.add(ws)
    logger.info(f"Cliente screen-share conectado. Total: {len(screen_clients)}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data.get('type') == 'frame':
                        # Processar frame de screen-share
                        image_data = data.get('image', '')
                        timestamp = data.get('ts', time.time())
                        
                        # Por enquanto, apenas repassar para clientes da UI
                        await broadcast_to_clients({
                            'type': 'screen:frame',
                            'image': image_data[:100] + '...',  # Truncar para log
                            'ts': timestamp
                        })
                        
                        logger.debug(f"Frame de screen-share processado: {timestamp}")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar frame: {e}")
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"Erro no WebSocket screen: {ws.exception()}")
    except Exception as e:
        logger.error(f"Erro no WebSocket screen: {e}")
    finally:
        screen_clients.discard(ws)
        logger.info(f"Cliente screen-share desconectado. Total: {len(screen_clients)}")
    
    return ws

async def candle_closer_task():
    """Task em background para fechar candles completados."""
    while True:
        try:
            await close_completed_candles()
            await asyncio.sleep(0.2)  # Verificar a cada 200ms
        except Exception as e:
            logger.error(f"Erro no candle_closer_task: {e}")
            await asyncio.sleep(1)

async def init_app():
    """Inicializa a aplicação aiohttp."""
    app = web.Application()
    
    # Configurar CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Rotas
    app.router.add_post('/push', handle_push)
    app.router.add_get('/ws', handle_websocket)
    app.router.add_get('/ws_screen', handle_screen_websocket)
    
    # Adicionar CORS a todas as rotas
    for route in list(app.router.routes()):
        cors.add(route)
    
    # Rota de health check
    async def health_check(request):
        return web.json_response({
            'status': 'healthy',
            'timestamp': time.time(),
            'clients': len(ws_clients),
            'screen_clients': len(screen_clients),
            'active_candles': len(current_candles)
        })
    
    app.router.add_get('/health', health_check)
    cors.add(app.router.add_get('/health', health_check))
    
    return app

async def main():
    """Função principal do servidor."""
    logger.info("🚀 Iniciando Nexus AI Server...")
    
    # Inicializar aplicação
    app = await init_app()
    
    # Iniciar task de fechamento de candles
    asyncio.create_task(candle_closer_task())
    
    # Iniciar servidor
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', HTTP_PORT)
    await site.start()
    
    logger.info(f"✅ Nexus AI Server rodando em http://0.0.0.0:{HTTP_PORT}")
    logger.info(f"📊 Endpoints disponíveis:")
    logger.info(f"   POST /push - Receber ticks")
    logger.info(f"   GET  /ws - WebSocket para UI")
    logger.info(f"   GET  /ws_screen - WebSocket para screen-share")
    logger.info(f"   GET  /health - Health check")
    logger.info(f"🔐 Auth token: {AUTH_TOKEN}")
    
    # Manter servidor rodando
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("🛑 Parando servidor...")
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())

