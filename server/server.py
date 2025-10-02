import os
import asyncio
import json
import logging
from pathlib import Path
from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_PUBLIC = PROJECT_ROOT / "web" / "ui" / "public"

# --- DEBUG / DIAGNOSTIC: mostrar onde o servidor procura a UI ---
logger.info("PROJECT_ROOT = %s", PROJECT_ROOT)
logger.info("UI_PUBLIC     = %s", UI_PUBLIC)
logger.info("UI_PUBLIC.exists() = %s", UI_PUBLIC.exists())
if UI_PUBLIC.exists():
    try:
        contents = [p.name for p in UI_PUBLIC.iterdir()]
        logger.info("UI_PUBLIC contents: %s", contents)
    except Exception:
        logger.exception("Erro listando UI_PUBLIC")
else:
    alt = Path.cwd() / "web" / "ui" / "public"
    logger.info("Alternative candidate (cwd) = %s , exists=%s", alt, alt.exists())

async def handle_index(request):
    return web.HTTPFound('/ui/')

async def handle_login(request):
    return web.HTTPFound('/ui/')

def create_app():
    app = web.Application()

    # fallback para corrigir path errado
    global UI_PUBLIC
    if not UI_PUBLIC.exists():
        alt = Path.cwd() / "web" / "ui" / "public"
        if alt.exists():
            logger.info("UI_PUBLIC não encontrado em PROJECT_ROOT; usando alternativa: %s", alt)
            UI_PUBLIC = alt

    # Rotas principais
    app.router.add_get('/', handle_index)
    app.router.add_get('/login', handle_login)

    # Servir estáticos
    ui_static_dir = UI_PUBLIC / "static"
    if ui_static_dir.exists():
        app.router.add_static('/ui/static', path=str(ui_static_dir), show_index=False)

    # Servir toda a pasta UI
    if UI_PUBLIC.exists():
        app.router.add_static('/ui', path=str(UI_PUBLIC), show_index=True)

    return app

def main():
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    web.run_app(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
