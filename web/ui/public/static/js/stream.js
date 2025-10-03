// stream.js — AGORA USA A CÂMERA (getUserMedia) para testes em celular.

const btnShare = document.getElementById('btnShare');
const btnStop = document.getElementById('btnStop');
const localVideo = document.getElementById('localVideo');
const wsStatus = document.getElementById('wsStatus');
const logs = document.getElementById('logs');

let ws = null;
let stream = null;
let captureInterval = null;

function log(msg){
  logs.textContent = new Date().toISOString() + ' ' + msg + '\n' + logs.textContent;
}

function connectWS(){
  if(ws) return;
  // Conexão WSS/WS adaptável para Render/Local
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = protocol + '://' + location.host + '/ws_screen';
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  ws.onopen = ()=>{ wsStatus.textContent = 'conectado'; log('WS conectado ' + url); };
  ws.onmessage = (ev)=>{ log('← ' + ev.data); };
  ws.onclose = ()=>{ wsStatus.textContent = 'desconectado'; ws=null; log('WS desconectado'); };
  ws.onerror = (e)=>{ log('WS erro: ' + e); };
}

async function startShare(){
  try{
    connectWS();
    
    // 🛑 AJUSTE: MUDANÇA PARA CÂMERA
    // Para teste em celular, usamos a câmera (getUserMedia) em vez da tela (getDisplayMedia).
    stream = await navigator.mediaDevices.getUserMedia({video:true, audio:false});
    
    // CORREÇÃO: Trata se o usuário cancelou a permissão da câmera
    if (!stream) {
        log('Captura de mídia (câmera) cancelada ou falhou.');
        return; 
    }
    
    localVideo.srcObject = stream;
    log('Captura de câmera ok. Iniciando envio de frames.');

    const track = stream.getVideoTracks()[0];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const settings = track.getSettings();
    canvas.width = settings.width || 1280;
    canvas.height = settings.height || 720;

    captureInterval = setInterval(async ()=>{
      try{
        // Desenha o frame atual da câmera no canvas
        ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
        
        // Converte o canvas para base64 JPEG
        const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
        
        if(ws && ws.readyState === WebSocket.OPEN){
          // Envia o frame via WebSocket para o backend da IA
          ws.send(JSON.stringify({image: dataUrl, ts: Date.now()}));
          log('→ frame enviado size=' + dataUrl.length);
        }
      }catch(e){
        log('Erro capture: ' + e);
      }
    }, 1000); // 1 fps
  }catch(e){
    // Este catch pegará erros de permissão de câmera (ex: NotAllowedError)
    log('Erro ao iniciar captura: ' + e);
  }
}

function stopShare(){
  if(captureInterval) clearInterval(captureInterval);
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  if(localVideo) localVideo.srcObject = null;
  log('Compartilhamento parado');
}

btnShare.addEventListener('click', startShare);
btnStop.addEventListener('click', stopShare);

// Connect WS when dashboard opened
connectWS();
