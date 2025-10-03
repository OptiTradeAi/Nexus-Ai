# server.py — Corrigido
import os
import time
import asyncio
import json
import base64
import logging
from pathlib import Path
from datetime import datetime
from aiohttp import web
import aiohttp_cors

logger = logging.getLogger("nexus_server")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------
# CORREÇÃO DE CAMINHOS PARA O AMBIENTE DOCKER (RENDER)
# O Dockerfile copiou o repo inteiro para /app
# -----------------------------------------------------------
PROJECT_ROOT = Path("/app") 
UI_PUBLIC = PROJECT_ROOT / "web" / "ui" / "public"

# Ensure folders (relativos ao /app, onde server.py está rodando)
FRAMES_DIR = Path("/app") / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
UPLOADS_DIR = Path("/app") / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
# -----------------------------------------------------------

# Admin credentials (use env vars in Render)
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "password")

# Try to import transcribe_service if present
try:
    import transcribe_service as transcribe_service
except Exception:
    transcribe_service = None
    logger.warning("transcribe_service not available; upload_audio endpoints will be disabled.")

# Try to import core kaon engine if present
try:
    from core import kaon_engine as kaon_engine
except Exception:
    kaon_engine = None
    logger.info("core.kaon_engine not found (integration optional).")

# ---------- Helpers ----------
def is_logged_in(request):
    return request.cookies.get("nexus_user") is not None

# ---------- Handlers ----------

async def index_redirect(request):
    # redirect to login page
    return web.HTTPFound('/ui/index.html')

async def ui_index(request):
    path = UI_PUBLIC / "index.html"
    if path.exists():
        return web.FileResponse(path)
    return web.Response(text="index.html not found", status=404)

async def ui_dashboard(request):
    # require login cookie
    if not is_logged_in(request):
        return web.HTTPFound('/ui/index.html')
    path = UI_PUBLIC / "dashboard.html"
    if path.exists():
        return web.FileResponse(path)
    return web.Response(text="dashboard.html not found", status=404)

async def login_post(request):
    data = await request.post()
    username = data.get("username")
    password = data.get("password")
    if username == ADMIN_USER and password == ADMIN_PASS:
        resp = web.HTTPFound('/ui/dashboard.html')
        resp.set_cookie('nexus_user', username, httponly=True)
        return resp
    # on failure return to index (frontend may show error)
    return web.HTTPFound('/ui/index.html?error=1')

async def logout_get(request):
    resp = web.HTTPFound('/ui/index.html')
    resp.del_cookie('nexus_user')
    return resp

async def health_check(request):
    return web.json_response({"status": "ok", "time": datetime.utcnow().isoformat()})

# WebSocket handler for receiving screen frames (base64 JSON or binary images)
async def ws_screen_handler(request):
    ws = web.WebSocketResponse(max_msg_size=20 * 1024 * 1024)
    await ws.prepare(request)
    logger.info("🔌 WebSocket client connected to /ws_screen")

    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
                # expected payload with key 'image' (dataURL) or control messages
                if 'image' in payload:
                    img_b64 = payload['image']
                    # strip data:prefix if present
                    if ',' in img_b64:
                        img_b64 = img_b64.split(',', 1)[1]
                    data_bytes = base64.b64decode(img_b64)
                    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                    fn = FRAMES_DIR / f'frame_{ts}.jpg'
                    fn.write_bytes(data_bytes)
                    logger.info(f"📸 Frame saved {fn.name} ({len(data_bytes)} bytes)")
                    # process in background
                    if kaon_engine and hasattr(kaon_engine, 'analyze_frame_from_path'):
                        asyncio.create_task(kaon_engine.analyze_frame_from_path(str(fn)))
                    await ws.send_json({"status": "ok", "saved": fn.name})
                else:
                    # other control message
                    logger.info("📩 WS TEXT message: %s", payload)
                    await ws.send_json({"status": "received", "payload": payload})
            except Exception as e:
                logger.exception("Error processing WS TEXT msg")
                await ws.send_json({"status": "error", "error": str(e)})
        elif msg.type == web.WSMsgType.BINARY:
            try:
                data = msg.data
                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                fn = FRAMES_DIR / f'frame_{ts}.jpg'
                fn.write_bytes(data)
                logger.info(f"📸 Frame (binary) saved {fn.name} ({len(data)} bytes)")
                if kaon_engine and hasattr(kaon_engine, 'analyze_frame_from_path'):
                    asyncio.create_task(kaon_engine.analyze_frame_from_path(str(fn)))
                await ws.send_json({"status": "ok", "saved": fn.name})
            except Exception as e:
                logger.exception("Error saving binary frame")
                await ws.send_json({"status": "error": str(e)})
        elif msg.type == web.WSMsgType.ERROR:
            logger.error('WebSocket connection closed with exception %s', ws.exception())

    logger.info("❌ WebSocket client disconnected /ws_screen")
    return ws

# Simple API: inject a candle (for testing)
async def api_test_candle(request):
    try:
        data = await request.json()
        logger.info('📩 /api/test_candle received: %s', data)
        # If your kaon_engine has handle_tick, call it (adapt signature)
        if kaon_engine and hasattr(kaon_engine, 'handle_tick'):
            try:
                # call handle_tick in background
                asyncio.create_task(kaon_engine.handle_tick(data))
            except Exception as e:
                logger.exception("Error calling kaon_engine.handle_tick")
        return web.json_response({'ok': True})
    except Exception as e:
        logger.exception("Error in /api/test_candle")
        return web.json_response({'error': str(e)}, status=500)

# Upload audio endpoint (multipart/form-data field 'file') -> starts transcription
async def upload_audio_handler(request):
    if transcribe_service is None:
        return web.json_response({"error": "transcribe service not configured"}, status=500)
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != 'file':
        return web.json_response({"error": "field 'file' not found"}, status=400)
    filename = field.filename or f"upload_{int(time.time())}.bin"
    filename = os.path.basename(filename)
    out_path = UPLOADS_DIR / filename
    with open(out_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)
    logger.info("🔊 Audio uploaded: %s", out_path.name)
    # schedule transcription in background
    try:
        asyncio.create_task(transcribe_service.transcribe_file_async(out_path, language="pt"))
        return web.json_response({"filename": out_path.name, "status": "processing"})
    except Exception as e:
        logger.exception("Error scheduling transcription")
        return web.json_response({"error": str(e)}, status=500)

# Query transcription result
async def get_transcription_handler(request):
    if transcribe_service is None:
        return web.json_response({"error": "transcribe service not configured"}, status=500)
    name = request.match_info.get("name")
    entry = transcribe_service.TRANSCRIPTIONS.get(name)
    if not entry:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response(entry)

# List transcriptions
async def list_transcriptions(request):
    if transcribe_service is None:
        return web.json_response({"error": "transcribe service not configured"}, status=500)
    return web.json_response(transcribe_service.TRANSCRIPTIONS)

# ---------- App factory ----------
def create_app():
    app = web.Application()

    # Setup CORS (for routes we will add below)
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # ----------------------------------------------------------------------
    # CORREÇÃO PARA RESOLVER A LISTAGEM DE DIRETÓRIO ("Index of /")
    # ----------------------------------------------------------------------
    
    # Mapeia a URL /ui/ para a pasta UI_PUBLIC.
    # show_index=True garante que o aiohttp procure e sirva o index.html como padrão.
    if UI_PUBLIC.exists():
        logger.info(f"UI_PUBLIC path: {UI_PUBLIC}")
        app.router.add_static('/ui/', path=str(UI_PUBLIC), show_index=True)
    else:
        logger.error(f"UI_PUBLIC directory not found at {UI_PUBLIC}. UI will fail to load.")

    # A rota original (servir apenas /ui/static) pode ser mantida
    ui_static_dir = UI_PUBLIC / "static"
    if ui_static_dir.exists():
        app.router.add_static('/ui/static', path=str(ui_static_dir), show_index=False)
    # ----------------------------------------------------------------------

    # ---------------- UI routes (use resource.add_route to avoid HEAD conflicts) ----------------
    # Root redirect
    res = app.router.add_resource('/')
    route = res.add_route('GET', index_redirect)
    cors.add(route)

    # index.html (login page)
    res = app.router.add_resource('/ui/index.html')
    route = res.add_route('GET', ui_index)
    cors.add(route)

    # dashboard (protected: check cookie)
    res = app.router.add_resource('/ui/dashboard.html')
    route = res.add_route('GET', ui_dashboard)
    cors.add(route)

    # login POST
    res = app.router.add_resource('/login')
    route = res.add_route('POST', login_post)
    cors.add(route)

    # logout
    res = app.router.add_resource('/logout')
    route = res.add_route('GET', logout_get)
    cors.add(route)

    # health
    res = app.router.add_resource('/health')
    route = res.add_route('GET', health_check)
    cors.add(route)

    # WS screen (allow connections)
    res = app.router.add_resource('/ws_screen')
    route = res.add_route('GET', ws_screen_handler)
    cors.add(route)

    # alias: /ws_stream also -> same handler
    res = app.router.add_resource('/ws_stream')
    route = res.add_route('GET', ws_screen_handler)
    cors.add(route)

    # API test candle
    res = app.router.add_resource('/api/test_candle')
    route = res.add_route('POST', api_test_candle)
    cors.add(route)

    # Upload / transcriptions
    if transcribe_service is not None:
        res = app.router.add_resource('/upload_audio')
        route = res.add_route('POST', upload_audio_handler)
        cors.add(route)

        res = app.router.add_resource('/transcriptions/{name}')
        route = res.add_route('GET', get_transcription_handler)
        cors.add(route)

        res = app.router.add_resource('/transcriptions')
        route = res.add_route('GET', list_transcriptions)
        cors.add(route)

    return app

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting Nexus AI Server on port {port} (ADMIN_USER={ADMIN_USER})")
    web.run_app(create_app(), host="0.0.0.0", port=port)
