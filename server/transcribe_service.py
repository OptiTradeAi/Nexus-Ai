# transcribe_service.py
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("transcribe_service")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")  # tiny recommended for tests
TRANSCRIPTIONS = {}  # filename -> {'status':.., 'text':..., 'error':...}

_model = None
_model_lock = asyncio.Lock()

async def load_model():
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                logger.info(f"Loading Whisper model '{WHISPER_MODEL}' (may download first time)...")
                loop = asyncio.get_event_loop()
                def _load():
                    import whisper
                    return whisper.load_model(WHISPER_MODEL)
                _model = await loop.run_in_executor(None, _load)
                logger.info("Whisper model loaded.")
    return _model

async def transcribe_file_async(filepath, language=None):
    fname = os.path.basename(filepath)
    TRANSCRIPTIONS[fname] = {"status": "running", "text": None, "error": None, "started_at": datetime.utcnow().isoformat()}
    try:
        model = await load_model()
        loop = asyncio.get_event_loop()
        def _transcribe():
            opts = {}
            if language:
                opts['language'] = language
            return model.transcribe(str(filepath), **opts)
        logger.info(f"Starting transcription for {fname} ...")
        result = await loop.run_in_executor(None, _transcribe)
        text = result.get("text", "").strip()
        TRANSCRIPTIONS[fname]["status"] = "done"
        TRANSCRIPTIONS[fname]["text"] = text
        TRANSCRIPTIONS[fname]["finished_at"] = datetime.utcnow().isoformat()
        logger.info(f"Transcription done for {fname}, chars={len(text)}")
        return text
    except Exception as e:
        logger.exception("Transcription error")
        TRANSCRIPTIONS[fname]["status"] = "error"
        TRANSCRIPTIONS[fname]["error"] = str(e)
        return None
