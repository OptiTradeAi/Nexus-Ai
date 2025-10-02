/**
 * Nexus AI - Content Bridge Script
 * 
 * Este script atua como ponte entre a página da corretora e a extensão,
 * injetando o page_hook.js e retransmitindo mensagens para o background.js
 * 
 * Funcionalidades:
 * - Injeta page_hook.js na página
 * - Escuta postMessage da página
 * - Retransmite dados para background.js
 * - Gerencia estado da conexão
 * 
 * Autor: Manus AI
 * Data: 2025-09-29
 */

(function() {
    'use strict';
    
    console.log('[Nexus AI Content] Iniciando content bridge...');
    
    // Estado da extensão
    let isHookInjected = false;
    let messageCount = 0;
    let lastMessageTime = 0;
    
    // Função para injetar o page hook
    function injectPageHook() {
        try {
            const script = document.createElement('script');
            script.src = chrome.runtime.getURL('page_hook.js');
            script.onload = function() {
                console.log('[Nexus AI Content] ✅ Page hook injetado');
                script.remove();
            };
            script.onerror = function() {
                console.error('[Nexus AI Content] ❌ Falha ao injetar page hook');
            };
            
            // Injetar no head ou documentElement
            const target = document.head || document.documentElement;
            if (target) {
                target.appendChild(script);
            } else {
                console.error('[Nexus AI Content] Elemento de injeção não encontrado');
            }
        } catch (error) {
            console.error('[Nexus AI Content] Erro na injeção:', error);
        }
    }
    
    // Função para enviar mensagem para background
    function sendToBackground(type, payload) {
        try {
            chrome.runtime.sendMessage({
                type: type,
                payload: payload,
                timestamp: Date.now(),
                url: window.location.href
            });
        } catch (error) {
            console.error('[Nexus AI Content] Erro ao enviar para background:', error);
        }
    }
    
    // Função para processar mensagens da página
    function handlePageMessage(event) {
        // Verificar origem e estrutura da mensagem
        if (!event.data || !event.data.__NEXUS_HOOK__) {
            return;
        }
        
        const { type, payload } = event.data;
        const currentTime = Date.now();
        
        // Throttling para evitar spam
        if (currentTime - lastMessageTime < 10) { // Mínimo 10ms entre mensagens
            return;
        }
        lastMessageTime = currentTime;
        
        try {
            switch (type) {
                case 'hook_injected':
                    isHookInjected = true;
                    console.log('[Nexus AI Content] Hook confirmado como injetado');
                    sendToBackground('NEXUS_HOOK_READY', {
                        url: window.location.href,
                        timestamp: payload.timestamp
                    });
                    break;
                
                case 'hook_error':
                    console.error('[Nexus AI Content] Erro no hook:', payload.error);
                    sendToBackground('NEXUS_HOOK_ERROR', payload);
                    break;
                
                case 'websocket_message':
                    messageCount++;
                    
                    // Log periódico para monitoramento
                    if (messageCount % 100 === 0) {
                        console.log(`[Nexus AI Content] ${messageCount} mensagens processadas`);
                    }
                    
                    // Enviar dados para background
                    sendToBackground('NEXUS_WS_DATA', {
                        ...payload,
                        messageCount: messageCount
                    });
                    break;
                
                default:
                    console.warn('[Nexus AI Content] Tipo de mensagem desconhecido:', type);
            }
        } catch (error) {
            console.error('[Nexus AI Content] Erro ao processar mensagem:', error);
        }
    }
    
    // Função para verificar se estamos na página correta
    function isTargetPage() {
        const hostname = window.location.hostname;
        return hostname.includes('homebroker.com');
    }
    
    // Função de inicialização
    function initialize() {
        if (!isTargetPage()) {
            console.log('[Nexus AI Content] Não é uma página alvo, ignorando');
            return;
        }
        
        console.log('[Nexus AI Content] Página alvo detectada:', window.location.href);
        
        // Adicionar listener para mensagens da página
        window.addEventListener('message', handlePageMessage, false);
        
        // Injetar hook após DOM estar pronto
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', injectPageHook);
        } else {
            injectPageHook();
        }
        
        // Enviar status inicial
        sendToBackground('NEXUS_CONTENT_READY', {
            url: window.location.href,
            readyState: document.readyState,
            timestamp: new Date().toISOString()
        });
        
        console.log('[Nexus AI Content] ✅ Content bridge inicializado');
    }
    
    // Listener para mensagens do background (se necessário)
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        try {
            switch (request.type) {
                case 'NEXUS_STATUS_REQUEST':
                    sendResponse({
                        isHookInjected: isHookInjected,
                        messageCount: messageCount,
                        url: window.location.href,
                        timestamp: Date.now()
                    });
                    break;
                
                case 'NEXUS_REINJECT_HOOK':
                    console.log('[Nexus AI Content] Reinjetando hook...');
                    injectPageHook();
                    sendResponse({ success: true });
                    break;
                
                default:
                    console.warn('[Nexus AI Content] Comando desconhecido:', request.type);
            }
        } catch (error) {
            console.error('[Nexus AI Content] Erro ao processar comando:', error);
            sendResponse({ error: error.message });
        }
    });
    
    // Inicializar
    initialize();
    
    // Cleanup ao descarregar página
    window.addEventListener('beforeunload', function() {
        console.log('[Nexus AI Content] Limpando content bridge...');
        window.removeEventListener('message', handlePageMessage);
    });
    
})();

