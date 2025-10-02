import os
import asyncio
import logging
from aiohttp import web
import aiohttp_cors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Health Check ----------
async def health_check(request):
    return web.json_response({"status": "ok"})

# ---------- Inicialização ----------
async def init_app():
    app = web.Application()

    # Configuração do CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # 🚀 CORRIGIDO: adiciona rota /health sem conflito com HEAD
    resource = app.router.add_resource("/health")
    route = resource.add_route("GET", health_check)
    cors.add(route)

    return app

# ---------- Main ----------
async def main():
    app = await init_app()

    # Porta dinâmica: Render define a variável de ambiente PORT
    port = int(os.environ.get("PORT", 8080))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info(f"🚀 Nexus AI Server rodando na porta {port}...")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Servidor encerrado manualmente")
