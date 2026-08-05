"""Point d'entree unique pour l'analyse IA d'un contenu. Orchestre les pipelines
dans l'ordre decrit dans ARCHITECTURE_IA.md section 5.

Integration cote stream_backend (2026-08, cablee) : ReelService._enqueue_ai_analysis
(app/services/reel_service.py) declenche cette tache juste apres la publication
d'un reel, via `celery_app.send_task("ai.analyze_reel", ..., queue="ai")` — fire
and forget, ne bloque jamais la publication. Note historique : le commentaire
precedent renvoyait vers app/tasks/video_tasks.py::process_reel_video et
`reel.video_url`, tous deux du code mort jamais appele et desormais desynchronise
du schema reel (le champ video_url n'existe plus sur le modele Reel).

Moderation (2026-08, cf. moderation_pipeline.py) : chaque analyze_* calcule un
verdict a 3 paliers (auto_remove / limited / none) et l'applique reellement
en base via pgvector_store.apply_moderation_verdict — cree un Report (compte
systeme dedie, cf. config.ai_moderation_reporter_id) et change le statut du
contenu pour les paliers limited/auto_remove. Jamais de suppression definitive
(DELETE), seulement un changement de statut reversible par un admin.

Notification au createur (2026-08) : dans TOUS les cas (y compris tier="none",
rien detecte), le createur/organisateur/artiste recoit une notification
persistante sur le sort de son propre contenu, via la tache Celery existante
cote stream_backend `app.tasks.notification_tasks.send_persistent_notification`
— reutilisee telle quelle (aucun nouveau code cote backend pour l'envoi), avec
un NotificationType par issue (reel_analysis_cleared/limited/removed, cf.
app/db/postgres/models/notification.py cote stream_backend). Envoyee via le
meme broker RabbitMQ, pas d'import croise entre les deux codebases.

Post/Event/Concert (2026-08) : analyses plus legeres que analyze_reel — pas de
video/audio (ffmpeg/faster-whisper), juste une image telechargee depuis son URL
R2/S3 (image_pipeline.download_image, cf. commentaire sur cette fonction pour
pourquoi un reel n'en a pas besoin) + le texte deja fourni par l'appelant.
Reutilisent le meme NotificationType reel_analysis_* : pas de nouvelle valeur
d'enum cote backend pour rester dans le meme cycle de migration deja deploye,
le corps du message est generique ("contenu" au lieu de "reel").
"""

import asyncio

from app.core.config import settings
from app.pipelines import audio_pipeline, image_pipeline, moderation_pipeline, text_pipeline, video_pipeline
from app.vector import pgvector_store
from app.workers.celery_app import celery_app

CATEGORY_CONFIDENCE_THRESHOLD = 0.5

# NotificationType cote stream_backend (app/db/postgres/models/notification.py)
# — dupliques ici en simples constantes str (pas d'import croise entre les 2
# codebases, cf. principe deja pose partout dans ce fichier/repo). Partages
# par les 4 types de contenu : seuls les 3 valeurs reel_analysis_* existent
# cote backend (ajouter post_analysis_*/event_analysis_*/concert_analysis_*
# demanderait une migration enum supplementaire, pas justifie pour un simple
# libelle de notification).
_TIER_TO_NOTIFICATION_TYPE = {
    "none": "reel_analysis_cleared",
    "limited": "reel_analysis_limited",
    "auto_remove": "reel_analysis_removed",
}

# Libelle humain par type de contenu, injecte dans le corps du message
# generique ci-dessous (evite de dupliquer 3 phrases x 4 types).
_CONTENT_LABEL = {
    "reel": "reel",
    "post": "publication",
    "event": "evenement",
    "concert": "concert",
}


def _notification_body(content_type: str, tier: str) -> str:
    label = _CONTENT_LABEL[content_type]
    if tier == "none":
        return f"Votre {label} a ete verifie automatiquement, tout est en ordre."
    if tier == "limited":
        return (
            f"Votre {label} a ete signale par notre systeme de verification automatique "
            "et est en cours de revue par un moderateur. Il reste visible sur votre "
            "profil mais n'apparait pas dans les recommandations pour le moment."
        )
    return (
        f"Votre {label} a ete retire automatiquement suite a une verification "
        "de contenu. Vous pouvez contester cette decision depuis le support."
    )


def _notify_owner(content_type: str, content_id: str, owner_id: str | None, tier: str) -> None:
    """Fire-and-forget : ne fait jamais echouer analyze_* si le backend ou
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
                _notification_body(content_type, tier),
            ],
            kwargs={"ref_id": content_id, "ref_type": content_type},
            queue="notifications",
        )
    except Exception:
        pass


def _finalize(content_type: str, content_id: str, classification: dict, verdict: dict) -> dict:
    """Etapes communes de fin d'analyse, partagees par analyze_reel et les 3
    taches image-seule : application du verdict, marquage "done", notification.

    tier == "none" (cleared) declenche mark_cleared() : transition
    pending_review -> published (2026-08bis) -- sans ca, un contenu cree en
    pending_review (invisible de tous tant que non confirme, cf.
    stream_backend create_reel/create_post/create_event/create_concert)
    resterait bloque invisible indefiniment meme apres un verdict positif.
    """
    if verdict["tier"] != "none":
        asyncio.run(pgvector_store.apply_moderation_verdict(
            content_type, content_id, verdict, settings.ai_moderation_reporter_id,
        ))
    else:
        asyncio.run(pgvector_store.mark_cleared(content_type, content_id))

    asyncio.run(pgvector_store.mark_analysis_done(content_type, content_id))

    owner_id = asyncio.run(pgvector_store.get_content_owner(content_type, content_id))
    _notify_owner(content_type, content_id, owner_id, verdict["tier"])

    return {"classification": classification, "moderation": verdict}


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
        embedding = image_pipeline.embed_image(frames[0])
        asyncio.run(pgvector_store.save_embedding("reel", reel_id, embedding))
        frame_signals = moderation_pipeline.moderate_frame(frames[0])

    verdict = moderation_pipeline.build_verdict(classification, frame_signals)

    return _finalize("reel", reel_id, classification, verdict)


def _analyze_image_content(content_type: str, content_id: str, image_url: str, text: str) -> dict:
    """Analyse partagee par post/event/concert : pas de video/audio, juste une
    image distante (telechargee via image_pipeline.download_image) et un texte
    deja assemble par l'appelant (caption+body, ou titre+description).

    Si image_url est vide ou le telechargement echoue, l'analyse se poursuit
    quand meme sur le texte seul (frame_signals=None) — meme principe de
    degradation gracieuse que analyze_reel quand extract_keyframes echoue."""
    classification = text_pipeline.classify_content(description=text)

    if classification["category"] and classification["confidence"] >= CATEGORY_CONFIDENCE_THRESHOLD:
        asyncio.run(pgvector_store.save_category(content_type, content_id, classification["category"]))

    frame_signals = None
    if image_url:
        image_path = image_pipeline.download_image(image_url)
        if image_path:
            embedding = image_pipeline.embed_image(image_path)
            asyncio.run(pgvector_store.save_embedding(content_type, content_id, embedding))
            frame_signals = moderation_pipeline.moderate_frame(image_path)

    verdict = moderation_pipeline.build_verdict(classification, frame_signals)

    return _finalize(content_type, content_id, classification, verdict)


@celery_app.task(name="ai.analyze_post")
def analyze_post(post_id: str, image_url: str = "", text: str = "") -> dict:
    return _analyze_image_content("post", post_id, image_url, text)


@celery_app.task(name="ai.analyze_event")
def analyze_event(event_id: str, image_url: str = "", title: str = "", description: str = "") -> dict:
    # event_service.py envoie 4 args positionnels (id, image_url, title,
    # description) contrairement a analyze_post (3 args, texte deja assemble
    # par l'appelant) -- signature alignee sur l'appelant reel plutot que
    # standardisee arbitrairement.
    text = "\n".join(p for p in [title, description] if p)
    return _analyze_image_content("event", event_id, image_url, text)


@celery_app.task(name="ai.analyze_concert")
def analyze_concert(concert_id: str, image_url: str = "", title: str = "", description: str = "") -> dict:
    text = "\n".join(p for p in [title, description] if p)
    return _analyze_image_content("concert", concert_id, image_url, text)
