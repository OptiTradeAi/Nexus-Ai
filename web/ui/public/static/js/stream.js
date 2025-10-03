// stream.js — Corrigido: adicionado tratamento para usuário que cancela getDisplayMedia
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
    stream = await navigator.mediaDevices.getDisplayMedia({video:true, audio:false});
    
    // 🛑 CORREÇÃO CRÍTICA DO TypeError: Se o usuário cancelar, o stream é null.
    if (!stream) {
        log('Compartilhamento cancelado pelo usuário ou falhou.');
        return; 
    }
    
    localVideo.srcObject = stream;
    log('getDisplayMedia ok');

    const track = stream.getVideoTracks()[0];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const settings = track.getSettings();
    canvas.width = settings.width || 1280;
    canvas.height = settings.height || 720;

    captureInterval = setInterval(async ()=>{
      try{
        ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
        if(ws && ws.readyState === WebSocket.OPEN){
          ws.send(JSON.stringify({image: dataUrl, ts: Date.now()}));
          log('→ frame enviado size=' + dataUrl.length);
        }
      }catch(e){
        log('Erro capture: ' + e);
      }
    }, 1000); // 1 fps
  }catch(e){
    // Este catch deve ser o que registra o TypeError (o erro agora deve ser mais específico, ex: NotAllowedError)
    log('Erro ao iniciar compartilhamento: ' + e);
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
