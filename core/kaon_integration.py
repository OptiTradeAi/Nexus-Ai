# kaon_integration.py
# Helper that calls existing core.kaon_engine functions if they exist.
import logging
from PIL import Image
import pytesseract

logger = logging.getLogger("kaon_integration")

async def analyze_frame_from_path(path):
    """
    Reads image, runs OCR and tries to call core.kaon_engine.handle_image or handle_tick if available.
    This file does NOT overwrite your existing core/kaon_engine. It's a safe integration helper.
    """
    try:
        # OCR (basic)
        img = Image.open(path)
        try:
            text = pytesseract.image_to_string(img, lang='por')
        except Exception:
            text = pytesseract.image_to_string(img)
        logger.info(f"📄 OCR extracted (first 200 chars): {text[:200]}")

        # try to call existing engine
        try:
            from core import kaon_engine
            if hasattr(kaon_engine, 'handle_image'):
                # if your engine exposes an async handle_image(path, ocr_text)
                try:
                    return await kaon_engine.handle_image(path, ocr_text=text)
                except TypeError:
                    # maybe handle_image is sync
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, kaon_engine.handle_image, path, text)
            else:
                logger.info("core.kaon_engine exists but no handle_image() found. OCR text returned.")
                return {"ocr_text": text}
        except Exception as e:
            logger.info("core.kaon_engine not available or raised an error. Returning OCR text.")
            return {"ocr_text": text}
    except Exception as e:
        logger.exception("Error analyzing frame")
        return {"error": str(e)}
