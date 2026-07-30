"""Extraction FFmpeg (frames cles + piste audio) et detection d'objets YOLOv11n.

Point d'entree du pipeline complet pour un reel/extrait de live — orchestre
video_pipeline -> image_pipeline / audio_pipeline / text_pipeline, cf. le flux
decrit dans ARCHITECTURE_IA.md section 5.
"""

import subprocess
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.model_manager import model_manager


def extract_keyframes(video_path: str, count: int = 3) -> list[str]:
    """Retourne les chemins des frames extraites, ou [] si ffmpeg echoue
    (fichier non-video, corrompu, URL inaccessible...) — ne leve jamais,
    cf. principe de degradation gracieuse (README "Avant de coder une
    nouvelle fonctionnalite"). Constate en prod le 2026-07-30 : sans ce
    garde-fou, un CalledProcessError ici faisait planter toute la tache
    Celery analyze_reel avant meme d'atteindre son `if frames:` (tasks.py),
    qui gere pourtant deja explicitement le cas 0 frame.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="ai_frames_"))
    pattern = str(out_dir / "frame_%02d.jpg")
    result = subprocess.run(
        [
            # La virgule dans mod(n,30) doit etre echappee (\,) pour le parseur
            # de filtergraph ffmpeg, qui sinon la lit comme separateur d'option
            # et fait echouer la commande a chaque appel — pas de shell ici
            # (liste d'arguments), donc c'est ffmpeg qui doit voir le backslash,
            # pas un echappement shell.
            "ffmpeg", "-i", video_path, "-vf", r"select='not(mod(n\,30))'",
            "-frames:v", str(count), "-vsync", "vfr", pattern, "-y",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    return sorted(str(p) for p in out_dir.glob("frame_*.jpg"))


def extract_audio(video_path: str) -> str | None:
    out_path = str(Path(tempfile.mkdtemp(prefix="ai_audio_")) / "audio.wav")
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", out_path, "-y"],
        capture_output=True,
    )
    if result.returncode != 0 or not Path(out_path).exists():
        return None
    return out_path


def _load_yolo():
    from ultralytics import YOLO

    return YOLO(settings.yolo_model_path)


def detect_objects(frame_path: str, confidence_threshold: float = 0.4) -> list[str]:
    model = model_manager.get("yolo", _load_yolo)
    results = model.predict(frame_path, verbose=False)

    labels: set[str] = set()
    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) >= confidence_threshold:
                labels.add(result.names[int(box.cls[0])])
    return sorted(labels)


# Sous-ensemble des 80 classes COCO (jeu d'entrainement standard de
# yolov11n.pt) pertinent pour un signal "objet sensible" — cf.
# moderation_pipeline.py. COCO ne contient PAS de classe "gun"/"weapon"/"blood" :
# yolov11n seul ne peut donc pas detecter une arme a feu ou une scene
# sanglante, uniquement "knife"/"scissors" (armes blanches/tranchants) qui y
# figurent reellement. Documente ici pour eviter toute fausse promesse de
# detection — cf. ARCHITECTURE_IA.md section 9 pour la discussion et les
# options (fine-tuning ou modele tiers) si ce signal s'avere insuffisant.
SENSITIVE_OBJECT_LABELS = {"knife", "scissors"}


def detect_sensitive_objects(frame_path: str, confidence_threshold: float = 0.4) -> list[str]:
    """Retourne le sous-ensemble de `detect_objects()` present dans SENSITIVE_OBJECT_LABELS."""
    labels = detect_objects(frame_path, confidence_threshold=confidence_threshold)
    return sorted(SENSITIVE_OBJECT_LABELS & set(labels))
