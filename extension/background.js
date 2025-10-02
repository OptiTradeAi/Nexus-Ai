/**
 * Nexus AI - Background Service Worker
 * 
 * Este service worker gerencia a comunicação entre a extensão e o servidor Nexus AI,
 * processando dados capturados e mantendo estado da conexão.
 * 
 * Funcionalidades:
 * - Recebe dados do content script
 * - Envia dados para servidor Nexus AI
 * - Gerencia configurações da extensão
 * - Monitora status da conexão
 * - Implementa retry logic
 * 
 * Autor: Manus AI
 * Data: 2025-09-29
 */

// Configurações globais
const CONFIG = {
    SERVER_URL: 'http://127.0.0.1:9000',  // URL local padrão
    PUSH_ENDPOINT: '/push',
    AUTH_TOKEN: 'nexus_dev_token_2025',
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000,
    BATCH_SIZE: 10,
    BATCH_TIMEOUT: 500
};

// Estado global
let serverUrl = CONFIG.SERVER_URL;
let authToken = CONFIG.AUTH_TOKEN;
let isConnected = false;
let messageQueue = [];
let batchTimer = null;
let stats = {
    messagesSent: 0,
    messagesQueued: 0,
    errors: 0,
    lastError: null,
    lastSuccess: null
};

console.log('[Nexus AI Background] 🚀 Service worker iniciado');

// Função para carregar configurações
async function loadConfig() {
    try {
        const result = await chrome.storage.sync.get([
            'serverUrl', 
            'authToken', 
            'enableBatching',
            'batchSize'
        ]);
        
        if (result.serverUrl) {
            serverUrl = result.serverUrl;
        }
        if (result.authToken) {
            authToken = result.authToken;
        }
        
        console.log('[Nexus AI Background] Configurações carregadas:', {
            serverUrl: serverUrl,
            hasToken: !!authToken
        });
    } catch (error) {
        console.error('[Nexus AI Background] Erro ao carregar configurações:', error);
    }
}

// Função para salvar configurações
async function saveConfig() {
    try {
        await chrome.storage.sync.set({
            serverUrl: serverUrl,
            authToken: authToken,
            lastUpdate: Date.now()
        });
    } catch (error) {
        console.error('[Nexus AI Background] Erro ao salvar configurações:', error);
    }
}

// Função para enviar dados para o servidor
async function sendToServer(payload, retryCount = 0) {
    try {
        const url = `${serverUrl}${CONFIG.PUSH_ENDPOINT}`;
        const headers = {
            'Content-Type': 'application/json'
        };
        
        // Adicionar token de autenticação se disponível
        if (authToken) {
            headers['Authorization'] = `Bearer ${authToken}`;
        }
        
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                forwarded: payload,
                source: 'nexus_extension',
                timestamp: Date.now(),
                version: '1.0.0'
            })
        });
        
        if (response.ok) {
            const responseText = await response.text();
            stats.messagesSent++;
            stats.lastSuccess = Date.now();
            isConnected = true;
            
            console.log('[Nexus AI Background] ✅ Dados enviados:', responseText);
            return true;
        } else {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
    } catch (error) {
        stats.errors++;
        stats.lastError = {
            message: error.message,
            timestamp: Date.now()
        };
        
        console.error('[Nexus AI Background] ❌ Erro ao enviar dados:', error);
        
        // Retry logic
        if (retryCount < CONFIG.RETRY_ATTEMPTS) {
            console.log(`[Nexus AI Background] 🔄 Tentativa ${retryCount + 1}/${CONFIG.RETRY_ATTEMPTS}`);
            await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY * (retryCount + 1)));
            return sendToServer(payload, retryCount + 1);
        }
        
        isConnected = false;
        return false;
    }
}

// Função para processar fila de mensagens em lote
function processBatch() {
    if (messageQueue.length === 0) return;
    
    const batch = messageQueue.splice(0, CONFIG.BATCH_SIZE);
    stats.messagesQueued -= batch.length;
    
    // Enviar cada mensagem do lote
    batch.forEach(async (payload) => {
        await sendToServer(payload);
    });
    
    // Agendar próximo lote se houver mais mensagens
    if (messageQueue.length > 0) {
        batchTimer = setTimeout(processBatch, CONFIG.BATCH_TIMEOUT);
    } else {
        batchTimer = null;
    }
}

// Função para adicionar mensagem à fila
function queueMessage(payload) {
    messageQueue.push(payload);
    stats.messagesQueued++;
    
    // Iniciar processamento em lote se não estiver rodando
    if (!batchTimer) {
        batchTimer = setTimeout(processBatch, CONFIG.BATCH_TIMEOUT);
    }
    
    // Processar imediatamente se a fila estiver cheia
    if (messageQueue.length >= CONFIG.BATCH_SIZE) {
        clearTimeout(batchTimer);
        processBatch();
    }
}

// Listener para mensagens dos content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    try {
        const { type, payload } = request;
        
        switch (type) {
            case 'NEXUS_WS_DATA':
                // Dados de WebSocket capturados
                console.log('[Nexus AI Background] 📊 Dados recebidos:', payload?.parsed?.event || 'unknown');
                queueMessage(payload);
                sendResponse({ success: true, queued: messageQueue.length });
                break;
            
            case 'NEXUS_HOOK_READY':
                console.log('[Nexus AI Background] ✅ Hook pronto na página:', payload.url);
                sendResponse({ success: true });
                break;
            
            case 'NEXUS_HOOK_ERROR':
                console.error('[Nexus AI Background] ❌ Erro no hook:', payload.error);
                sendResponse({ success: false, error: payload.error });
                break;
            
            case 'NEXUS_CONTENT_READY':
                console.log('[Nexus AI Background] ✅ Content script pronto:', payload.url);
                sendResponse({ success: true });
                break;
            
            case 'NEXUS_GET_STATUS':
                // Retornar status da extensão
                sendResponse({
                    isConnected: isConnected,
                    serverUrl: serverUrl,
                    stats: stats,
                    queueLength: messageQueue.length
                });
                break;
            
            case 'NEXUS_UPDATE_CONFIG':
                // Atualizar configurações
                if (payload.serverUrl) serverUrl = payload.serverUrl;
                if (payload.authToken) authToken = payload.authToken;
                saveConfig();
                sendResponse({ success: true });
                break;
            
            default:
                console.warn('[Nexus AI Background] Tipo de mensagem desconhecido:', type);
                sendResponse({ success: false, error: 'Unknown message type' });
        }
    } catch (error) {
        console.error('[Nexus AI Background] Erro ao processar mensagem:', error);
        sendResponse({ success: false, error: error.message });
    }
    
    return true; // Manter canal aberto para resposta assíncrona
});

// Listener para instalação/atualização da extensão
chrome.runtime.onInstalled.addListener((details) => {
    console.log('[Nexus AI Background] Extensão instalada/atualizada:', details.reason);
    
    if (details.reason === 'install') {
        // Primeira instalação
        console.log('[Nexus AI Background] 🎉 Primeira instalação - configurando...');
        saveConfig();
    } else if (details.reason === 'update') {
        // Atualização
        console.log('[Nexus AI Background] 🔄 Extensão atualizada');
        loadConfig();
    }
});

// Listener para inicialização do service worker
chrome.runtime.onStartup.addListener(() => {
    console.log('[Nexus AI Background] 🔄 Service worker reiniciado');
    loadConfig();
});

// Função para testar conexão com servidor
async function testConnection() {
    try {
        const response = await fetch(`${serverUrl}/health`);
        if (response.ok) {
            isConnected = true;
            console.log('[Nexus AI Background] ✅ Conexão com servidor OK');
        } else {
            isConnected = false;
            console.warn('[Nexus AI Background] ⚠️ Servidor respondeu com erro:', response.status);
        }
    } catch (error) {
        isConnected = false;
        console.error('[Nexus AI Background] ❌ Falha na conexão com servidor:', error.message);
    }
}

// Função para monitoramento periódico
function startMonitoring() {
    // Testar conexão a cada 30 segundos
    setInterval(testConnection, 30000);
    
    // Log de estatísticas a cada 5 minutos
    setInterval(() => {
        console.log('[Nexus AI Background] 📊 Estatísticas:', {
            ...stats,
            queueLength: messageQueue.length,
            isConnected: isConnected
        });
    }, 300000);
}

// Inicialização
async function initialize() {
    await loadConfig();
    await testConnection();
    startMonitoring();
    
    console.log('[Nexus AI Background] ✅ Service worker inicializado');
}

// Inicializar quando o service worker for carregado
initialize();

