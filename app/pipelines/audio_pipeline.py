"""Transcription audio via faster-whisper (CTranslate2 — plus leger que Whisper original en CPU).

Voir ARCHITECTURE_IA.md section 3 : ne pas transcrire un live entier en continu,
seulement des extraits periodiques (quelques dizaines de secondes) pour rester
dans le budget CPU.
"""

from app.core.config import settings
from app.core.model_manager import model_manager


def _load_whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")


def transcribe(audio_path: str) -> str:
    model = model_manager.get("whisper", _load_whisper)
    segments, _info = model.transcribe(audio_path, beam_size=1)
    return " ".join(segment.text.strip() for segment in segments)
