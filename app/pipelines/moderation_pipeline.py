"""Fusionne les signaux de moderation (texte, image NSFW, objets sensibles)
en une decision unique.

Ne prend AUCUNE action de moderation lui-meme (pas de suppression, pas
d'ecriture DB) : produit uniquement un verdict structure, cf.
ARCHITECTURE_IA.md section 9. C'est l'appelant (workers/tasks.py) qui decide
quoi en faire (flag pour revue humaine, cf. decision produit du 2026-07-30 —
jamais de depublication automatique).

Design volontairement conservateur : le but est de faire remonter des
candidats a la revue humaine, pas de juger. Un faux negatif (contenu limite
non detecte) reste modere par le systeme de signalement utilisateur existant
(stream_backend/app/services/report_service.py) ; un faux positif genere une
revue manuelle inutile mais ne supprime jamais rien tout seul.
"""

from app.pipelines import video_pipeline
from app.pipelines.image_pipeline import classify_nsfw

# Seuils de confiance au-dela desquels un signal individuel declenche un flag.
# Volontairement hauts (peu de faux positifs) car aucune revue humaine
# n'existe encore en amont — cf. ARCHITECTURE_IA.md section 9.
#
# NSFW_CONFIDENCE_THRESHOLD releve a 0.8 suite a un faux positif reel constate
# le 2026-07-30 : une photo de couteau de cuisine (aucun contenu sexuel)
# classee "sexual_explicit" a 0.6084 avec l'ancienne temperature de softmax
# (100). Meme apres correction de la temperature (image_pipeline.py, 100->20),
# `confidence` sur ce classifieur zero-shot CLIP n'est PAS une probabilite
# calibree — un seuil haut reste necessaire pour limiter les faux positifs
# tant qu'aucune revue humaine n'absorbe le signal en amont.
NSFW_CONFIDENCE_THRESHOLD = 0.8
TEXT_FLAG_CONFIDENCE_THRESHOLD = 0.5

# Correspondance vers ReportReason (stream_backend/app/db/postgres/models/report.py)
# — permet a l'appelant de creer un Report avec la meme taxonomie sans mapping
# supplementaire. "suggestive" (CLIP) ne mappe sur rien ici : signal trop faible
# pour justifier une revue humaine a lui seul (cf. NSFW_CONFIDENCE_THRESHOLD).
_NSFW_LABEL_TO_REPORT_REASON = {"sexual_explicit": "inappropriate"}
_TEXT_FLAG_REASON_TO_REPORT_REASON = {
    "spam": "spam",
    "arnaque": "spam",
    "haine": "harassment",
    "violence": "violence",
}
_SENSITIVE_OBJECT_REPORT_REASON = "violence"


def moderate_frame(frame_path: str) -> dict:
    """Signal image seul : NSFW (CLIP zero-shot) + objets sensibles (YOLO).

    Retourne {"nsfw": {...}, "sensitive_objects": [...]}."""
    nsfw = classify_nsfw(frame_path)
    sensitive_objects = video_pipeline.detect_sensitive_objects(frame_path)
    return {"nsfw": nsfw, "sensitive_objects": sensitive_objects}


def build_verdict(
    text_classification: dict,
    frame_signals: dict | None = None,
) -> dict:
    """Combine tous les signaux disponibles en un verdict unique.

    `text_classification` : sortie de text_pipeline.classify_content().
    `frame_signals` : sortie de moderate_frame(), ou None si pas de frame
    exploitable (ex. post texte seul).

    Retourne {"should_flag": bool, "reasons": list[str], "signals": {...}} —
    `reasons` utilise deja la taxonomie ReportReason de stream_backend, pret
    a etre passe tel quel a la creation d'un Report (voir docstring module).
    Plusieurs raisons possibles : chaque signal au-dessus de son seuil ajoute
    sa raison, aucune n'ecrase les autres.
    """
    reasons: list[str] = []

    if text_classification.get("flagged"):
        flag_reason = text_classification.get("flag_reason")
        confidence = text_classification.get("confidence", 0.0)
        if confidence >= TEXT_FLAG_CONFIDENCE_THRESHOLD:
            mapped = _TEXT_FLAG_REASON_TO_REPORT_REASON.get(flag_reason)
            if mapped:
                reasons.append(mapped)

    if frame_signals:
        nsfw = frame_signals.get("nsfw") or {}
        if (
            nsfw.get("label") in _NSFW_LABEL_TO_REPORT_REASON
            and nsfw.get("confidence", 0.0) >= NSFW_CONFIDENCE_THRESHOLD
        ):
            reasons.append(_NSFW_LABEL_TO_REPORT_REASON[nsfw["label"]])

        if frame_signals.get("sensitive_objects"):
            reasons.append(_SENSITIVE_OBJECT_REPORT_REASON)

    # Deduplique en gardant l'ordre d'apparition (stable pour les tests/logs)
    seen: set[str] = set()
    unique_reasons = [r for r in reasons if not (r in seen or seen.add(r))]

    return {
        "should_flag": bool(unique_reasons),
        "reasons": unique_reasons,
        "signals": {
            "text": text_classification,
            "frame": frame_signals,
        },
    }
