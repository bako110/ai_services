"""Point d'entree unique pour l'analyse IA d'un contenu. Orchestre les pipelines
dans l'ordre decrit dans ARCHITECTURE_IA.md section 5.

Integration cote stream_backend (2026-08, cablee) : ReelService._enqueue_ai_analysis
(app/services/reel_service.py) declenche cette tache juste apres la publication
d'un reel, via `celery_app.send_task("ai.analyze_reel", ..., queue="ai")` — fire
and forget, ne bloque jamais la publication. Note historique : le commentaire
precedent renvoyait vers app/tasks/video_tasks.py::process_reel_video et
`reel.video_url`, tous deux du code mort jamais appele et desormais desynchronise
du schema reel (le champ video_url n'existe plus sur le modele Reel).

Moderation (2026-08, cf. moderation_pipeline.py) : `analyze_reel` calcule un
verdict a 3 paliers (auto_remove / limited / none) et l'applique reellement
en base via pgvector_store.apply_moderation_verdict — cree un Report (compte
systeme dedie, cf. config.ai_moderation_reporter_id) et change le statut du
reel pour les paliers limited/auto_remove. Jamais de suppression definitive
(DELETE), seulement un changement de statut reversible par un admin.

Notification au createur (2026-08) : dans TOUS les cas (y compris tier="none",
rien detecte), le createur recoit une notification persistante sur le sort de
son propre reel, via la tache Celery existante cote stream_backend
`app.tasks.notification_tasks.send_persistent_notification` — reutilisee telle
quelle (aucun nouveau code cote backend pour l'envoi), juste un nouveau
NotificationType par issue (reel_analysis_cleared/limited/removed, cf.
app/db/postgres/models/notification.py cote stream_backend). Envoyee via le
meme broker RabbitMQ (send_task vers la queue "default" du backend, pas
d'import croise entre les deux codebases — meme pattern que le declenchement
inverse stream_backend -> ai_service).
"""

import asyncio

from app.core.config import settings
from app.pipelines import audio_pipeline, moderation_pipeline, text_pipeline, video_pipeline
from app.vector import pgvector_store
from app.workers.celery_app import celery_app

CATEGORY_CONFIDENCE_THRESHOLD = 0.5

# NotificationType cote stream_backend (app/db/postgres/models/notification.py)
# — dupliques ici en simples constantes str (pas d'import croise entre les 2
# codebases, cf. principe deja pose partout dans ce fichier/repo).
_TIER_TO_NOTIFICATION_TYPE = {
    "none": "reel_analysis_cleared",
    "limited": "reel_analysis_limited",
    "auto_remove": "reel_analysis_removed",
}
_TIER_TO_NOTIFICATION_BODY = {
    "none": "Votre reel a ete verifie automatiquement, tout est en ordre.",
    "limited": (
        "Votre reel a ete signale par notre systeme de verification automatique "
        "et est en cours de revue par un moderateur. Il reste visible sur votre "
        "profil mais n'apparait pas dans les recommandations pour le moment."
    ),
    "auto_remove": (
        "Votre reel a ete retire automatiquement suite a une verification "
        "de contenu. Vous pouvez contester cette decision depuis le support."
    ),
}


def _notify_owner(reel_id: str, owner_id: str | None, tier: str) -> None:
    """Fire-and-forget : ne fait jamais echouer analyze_reel si le backend ou
    RabbitMQ est indisponible au moment de l'envoi.

    queue="notifications", pas "default" : constate en prod (2026-08) que
    app.tasks.notification_tasks.* est routee vers la queue "notifications"
    cote stream_backend (task_routes, app/tasks/celery_app.py), qui n'a
    aucun argument special declare (pas de TTL/DLQ contrairement a "default").
    Envoyer sur "default" avec les mauvais arguments de queue faisait
    echouer la publication RabbitMQ : PRECONDITION_FAILED
    "inequivalent arg 'x-message-ttl'" (la queue "default" existante cote
    backend a un TTL de 24h + DLQ que kombu tentait de re-declarer sans, cf.
    kombu/RabbitMQ qui exige des arguments identiques a chaque (re)declaration
    d'une queue existante)."""
    if not owner_id:
        return
    try:
        celery_app.send_task(
            "app.tasks.notification_tasks.send_persistent_notification",
            args=[
                owner_id,
                _TIER_TO_NOTIFICATION_TYPE[tier],
                _TIER_TO_NOTIFICATION_BODY[tier],
            ],
            kwargs={"ref_id": reel_id, "ref_type": "reel"},
            queue="notifications",
        )
    except Exception:
        pass


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

    asyncio.run(pgvector_store.mark_analysis_done("reel", reel_id))

    owner_id = asyncio.run(pgvector_store.get_content_owner("reel", reel_id))
    _notify_owner(reel_id, owner_id, verdict["tier"])

    return {"classification": classification, "moderation": verdict}
