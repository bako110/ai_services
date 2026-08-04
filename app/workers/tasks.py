"""Point d'entree unique pour l'analyse IA d'un contenu. Orchestre les pipelines
dans l'ordre decrit dans ARCHITECTURE_IA.md section 5.

Integration cote stream_backend : app/tasks/video_tasks.py::process_reel_video
est le point naturel pour declencher cette tache une fois le reel publie.
Comme ai_service partage le MEME RabbitMQ que stream_backend (voir
app/workers/celery_app.py), stream_backend peut l'enfiler sans importer le
code d'ai_service, juste par nom de tache :

    from app.tasks.celery_app import celery_app as backend_celery_app
    backend_celery_app.send_task(
        "ai.analyze_reel",
        args=[str(reel.id), reel.video_url, reel.caption or ""],
        queue="ai",
    )

A ajouter dans process_reel_video() apres reel.status = ReelStatus.published,
uniquement si reel.category est encore None (ne jamais ecraser un choix du
createur, cf. save_category() qui filtre deja `WHERE category IS NULL`).

Moderation (2026-08, cf. moderation_pipeline.py) : `analyze_reel` calcule un
verdict a 3 paliers (auto_remove / limited / none) et l'applique reellement
en base via pgvector_store.apply_moderation_verdict — cree un Report (compte
systeme dedie, cf. config.ai_moderation_reporter_id) et change le statut du
reel pour les paliers limited/auto_remove. Jamais de suppression definitive
(DELETE), seulement un changement de statut reversible par un admin.
"""

import asyncio

from app.core.config import settings
from app.pipelines import audio_pipeline, moderation_pipeline, text_pipeline, video_pipeline
from app.vector import pgvector_store
from app.workers.celery_app import celery_app

CATEGORY_CONFIDENCE_THRESHOLD = 0.5


@celery_app.task(name="ai.analyze_reel")
def analyze_reel(reel_id: str, video_path: str, title: str = "", description: str = "") -> dict:
    frames = video_pipeline.extract_keyframes(video_path)
    audio_path = video_pipeline.extract_audio(video_path)

    transcription = audio_pipeline.transcribe(audio_path) if audio_path else ""
    detected_objects = video_pipeline.detect_objects(frames[0]) if frames else []

    classification = text_pipeline.classify_content(
        title=title,
        description=description,
        transcription=transcription,
        detected_objects=detected_objects,
    )

    if classification["category"] and classification["confidence"] >= CATEGORY_CONFIDENCE_THRESHOLD:
        asyncio.run(pgvector_store.save_category("reel", reel_id, classification["category"]))

    frame_signals = None
    if frames:
        from app.pipelines import image_pipeline

        embedding = image_pipeline.embed_image(frames[0])
        asyncio.run(pgvector_store.save_embedding("reel", reel_id, embedding))
        frame_signals = moderation_pipeline.moderate_frame(frames[0])

    verdict = moderation_pipeline.build_verdict(classification, frame_signals)

    if verdict["tier"] != "none":
        asyncio.run(pgvector_store.apply_moderation_verdict(
            "reel", reel_id, verdict, settings.ai_moderation_reporter_id,
        ))

    return {"classification": classification, "moderation": verdict}
