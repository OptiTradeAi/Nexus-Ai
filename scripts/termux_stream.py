#!/usr/bin/env python3
# Termux stream -> captura screenshot e envia para /ws_screen (base64)
import os, time, base64, json, subprocess
from websocket import create_connection

WS_URL = os.environ.get('NEXUS_WS','wss://nexus-ai-us1s.onrender.com/ws_screen')
print("Conectando em", WS_URL)
ws = create_connection(WS_URL)

try:
    while True:
        # caminho temporário no sdcard
        out = '/sdcard/screen.png'
        # comando nativo Android
        subprocess.run(['screencap', '-p', out], check=True)
        with open(out, 'rb') as f:
            b = f.read()
        b64 = base64.b64encode(b).decode()
        payload = json.dumps({'type':'frame', 'image': 'data:image/png;base64,'+b64, 'ts': time.time()})
        ws.send(payload)
        print("frame enviado", len(b))
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    ws.close()
