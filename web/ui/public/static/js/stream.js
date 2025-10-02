// stream.js — captura a tela (getDisplayMedia) e envia frames via WebSocket para /ws_screen
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
