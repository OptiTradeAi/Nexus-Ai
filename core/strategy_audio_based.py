# strategy_audio_based.py
# Esqueleto: regras extra baseadas na transcrição do áudio.
async def evaluate(candle, context=None):
    """
    Recebe um candle {open,high,low,close,time} e retorna:
    {'action': 'buy'|'sell'|'hold', 'confidence': float}
    Substitua/complete as regras aqui quando tiver a transcrição do áudio.
    """
    o = candle.get('open')
    c = candle.get('close')
    if o is None or c is None:
        return {'action':'hold','confidence':0.0}
    if c > o * 1.001:
        return {'action':'buy','confidence':0.6}
    return {'action':'hold','confidence':0.2}
