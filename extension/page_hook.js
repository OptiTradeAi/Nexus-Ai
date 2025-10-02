/**
 * Nexus AI - Page Hook Script
 * 
 * Este script é injetado na página da corretora para interceptar
 * o tráfego WebSocket e capturar ticks em tempo real.
 * 
 * Funcionalidades:
 * - Intercepta WebSocket.constructor
 * - Filtra mensagens relevantes do Pusher
 * - Envia dados para a extensão via postMessage
 * - Sanitiza dados sensíveis automaticamente
 * 
 * Autor: Manus AI
 * Data: 2025-09-29
 */

(function() {
    'use strict';
    
    try {
        console.log('[Nexus AI] Iniciando interceptação de WebSocket...');
        
        // Backup do WebSocket original
        const OriginalWebSocket = window.WebSocket;
        if (!OriginalWebSocket) {
            console.warn('[Nexus AI] WebSocket não disponível');
            return;
        }
        
        // Função para tentar parsear JSON
        function tryParseJSON(str) {
            try {
                return JSON.parse(str);
            } catch (e) {
                return null;
            }
        }
        
        // Função para sanitizar dados sensíveis
        function sanitizeData(data) {
            if (typeof data !== 'string') return data;
            
            const sensitivePatterns = [
                /token["\s]*[:=]["\s]*[^",\s}]+/gi,
                /password["\s]*[:=]["\s]*[^",\s}]+/gi,
                /auth["\s]*[:=]["\s]*[^",\s}]+/gi,
                /key["\s]*[:=]["\s]*[^",\s}]+/gi,
                /secret["\s]*[:=]["\s]*[^",\s}]+/gi
            ];
            
            let sanitized = data;
            sensitivePatterns.forEach(pattern => {
                sanitized = sanitized.replace(pattern, (match) => {
                    const parts = match.split(/[:=]/);
                    return parts[0] + (match.includes(':') ? ':"[REDACTED]"' : '="[REDACTED]"');
                });
            });
            
            return sanitized;
        }
        
        // Função para verificar se a mensagem é relevante
        function isRelevantMessage(url, parsed) {
            // Verificar se é do Pusher
            if (!url.includes('pusher.com')) return false;
            
            // Verificar se contém dados de trading
            if (parsed && typeof parsed === 'object') {
                const hasEvent = 'event' in parsed;
                const hasData = 'data' in parsed;
                const hasPrice = parsed.data && (
                    parsed.data.includes('price') || 
                    parsed.data.includes('last') ||
                    parsed.data.includes('timestamp')
                );
                
                return hasEvent && hasData && hasPrice;
            }
            
            return false;
        }
        
        // WebSocket interceptado
        function NexusWebSocket(url, protocols) {
            console.log('[Nexus AI] WebSocket criado:', url);
            
            // Criar WebSocket original
            const ws = protocols ? 
                new OriginalWebSocket(url, protocols) : 
                new OriginalWebSocket(url);
            
            // Interceptar mensagens recebidas
            ws.addEventListener('message', function(event) {
                try {
                    let data = event.data;
                    
                    // Converter ArrayBuffer para string se necessário
                    if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
                        try {
                            data = new TextDecoder().decode(data);
                        } catch (e) {
                            data = '[binary_data]';
                        }
                    }
                    
                    // Tentar parsear JSON
                    const parsed = (typeof data === 'string') ? tryParseJSON(data) : null;
                    
                    // Verificar se é mensagem relevante
                    if (isRelevantMessage(url, parsed)) {
                        // Sanitizar dados
                        const sanitizedData = sanitizeData(typeof data === 'string' ? data : '[binary_data]');
                        
                        // Criar payload para enviar
                        const payload = {
                            source: 'nexus_page_hook',
                            url: url,
                            raw: sanitizedData,
                            parsed: parsed,
                            timestamp: new Date().toISOString(),
                            user_agent: navigator.userAgent.substring(0, 50) // Apenas parte do user agent
                        };
                        
                        // Enviar via postMessage
                        window.postMessage({
                            __NEXUS_HOOK__: true,
                            type: 'websocket_message',
                            payload: payload
                        }, '*');
                        
                        console.log('[Nexus AI] Tick capturado:', parsed?.event || 'unknown');
                    }
                } catch (error) {
                    console.error('[Nexus AI] Erro ao processar mensagem:', error);
                }
            });
            
            // Interceptar mensagens enviadas (opcional, para debug)
            const originalSend = ws.send.bind(ws);
            ws.send = function(data) {
                try {
                    // Log de mensagens enviadas (sem dados sensíveis)
                    console.log('[Nexus AI] WebSocket send:', typeof data);
                    
                    // Enviar mensagem original
                    return originalSend(data);
                } catch (error) {
                    console.error('[Nexus AI] Erro ao enviar mensagem:', error);
                    throw error;
                }
            };
            
            // Interceptar eventos de conexão
            ws.addEventListener('open', function() {
                console.log('[Nexus AI] WebSocket conectado:', url);
            });
            
            ws.addEventListener('close', function() {
                console.log('[Nexus AI] WebSocket desconectado:', url);
            });
            
            ws.addEventListener('error', function(error) {
                console.error('[Nexus AI] Erro no WebSocket:', error);
            });
            
            return ws;
        }
        
        // Copiar propriedades estáticas
        NexusWebSocket.prototype = OriginalWebSocket.prototype;
        NexusWebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
        NexusWebSocket.OPEN = OriginalWebSocket.OPEN;
        NexusWebSocket.CLOSING = OriginalWebSocket.CLOSING;
        NexusWebSocket.CLOSED = OriginalWebSocket.CLOSED;
        
        // Substituir WebSocket global
        window.WebSocket = NexusWebSocket;
        
        console.log('[Nexus AI] ✅ WebSocket interceptado com sucesso');
        
        // Enviar confirmação de injeção
        window.postMessage({
            __NEXUS_HOOK__: true,
            type: 'hook_injected',
            payload: {
                timestamp: new Date().toISOString(),
                url: window.location.href
            }
        }, '*');
        
    } catch (error) {
        console.error('[Nexus AI] ❌ Falha na injeção do hook:', error);
        
        // Enviar erro via postMessage
        window.postMessage({
            __NEXUS_HOOK__: true,
            type: 'hook_error',
            payload: {
                error: error.message,
                timestamp: new Date().toISOString()
            }
        }, '*');
    }
})();

