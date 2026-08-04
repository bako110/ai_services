"""Fusionne les signaux de moderation (texte, image NSFW, objets sensibles)
en un verdict a 3 paliers, inspire du modele de moderation par paliers de
grandes plateformes (retrait auto / diffusion limitee / file moderateur
humain) — cf. discussion produit 2026-08.

Ne prend AUCUNE action de suppression definitive lui-meme : produit un
verdict structure que l'appelant (workers/tasks.py) applique (changement de
statut reel, creation de Report). Design volontairement conservateur : le
palier le plus severe (retrait automatique) exige un ACCORD ENTRE PLUSIEURS
SIGNAUX INDEPENDANTS plutot qu'un score de confiance eleve sur un seul
signal — les scores de confiance individuels se sont montres peu fiables en
pratique (cf. NSFW_CONFIDENCE_THRESHOLD ci-dessous, faux positif reel a
60% sur une photo de couteau de cuisine ; confidence texte fixee a 1.0 faute
de fiabilite du LLM a l'estimer). Compter les signaux qui s'accordent est
plus robuste que de faire confiance a un chiffre non calibre.

Un faux negatif (contenu limite non detecte) reste modere par le systeme de
signalement utilisateur existant (stream_backend/app/services/report_service.py) ;
un faux positif au palier "diffusion limitee" ne supprime jamais rien, un
faux positif au palier "retrait auto" reste possible mais nettement moins
probable qu'avec un seuil sur un seul signal.
"""

from app.pipelines import video_pipeline
from app.pipelines.image_pipeline import classify_nsfw

# Seuils de confiance au-dela desquels un signal individuel COMPTE comme
# "signal positif" dans le vote a plusieurs signaux ci-dessous — pas des
# seuils de decision finale a eux seuls.
NSFW_CONFIDENCE_THRESHOLD = 0.8
TEXT_FLAG_CONFIDENCE_THRESHOLD = 0.5

# Correspondance vers ReportReason (stream_backend/app/db/postgres/models/report.py)
# — permet a l'appelant de creer un Report avec la meme taxonomie sans mapping
# supplementaire. "suggestive" (CLIP) ne mappe sur rien ici : signal trop faible
# pour compter meme comme signal individuel (cf. NSFW_CONFIDENCE_THRESHOLD).
_NSFW_LABEL_TO_REPORT_REASON = {"sexual_explicit": "inappropriate"}
_TEXT_FLAG_REASON_TO_REPORT_REASON = {
    "spam": "spam",
    "arnaque": "spam",
    "haine": "harassment",
    "violence": "violence",
}
_SENSITIVE_OBJECT_REPORT_REASON = "violence"

# Paliers du verdict, du plus au moins severe.
TIER_AUTO_REMOVE = "auto_remove"      # >=2 signaux independants d'accord
TIER_LIMITED = "limited"              # exactement 1 signal positif
TIER_NONE = "none"                    # aucun signal


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
    """Combine tous les signaux disponibles en un verdict a 3 paliers.

    `text_classification` : sortie de text_pipeline.classify_content().
    `frame_signals` : sortie de moderate_frame(), ou None si pas de frame
    exploitable (ex. post texte seul).

    Retourne {"tier": str, "reasons": list[str], "signal_count": int,
    "signals": {...}} — `reasons` utilise deja la taxonomie ReportReason de
    stream_backend, pret a etre passe tel quel a la creation d'un Report.

    Logique des paliers, basee sur le NOMBRE de signaux independants
    positifs (texte, NSFW, objets sensibles — 3 sources au maximum), pas sur
    la confiance individuelle de chacun :
    - 0 signal positif -> TIER_NONE, rien ne se passe.
    - 1 signal positif -> TIER_LIMITED, diffusion limitee (exclu des feeds
      de decouverte, reste visible sur le profil/lien direct) + Report cree
      pour revue humaine.
    - 2+ signaux positifs (ex. texte ET image tous les deux suspects) ->
      TIER_AUTO_REMOVE, retrait immediat + Report cree et deja marque
      resolu (l'action a deja ete prise, pas juste en attente).
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

    # Deduplique en gardant l'ordre d'apparition (stable pour les tests/logs).
    # Note : un texte ET une image qui mappent tous les deux sur "violence"
    # comptent comme 1 seule raison unique dans `reasons`, mais comme 2
    # SIGNAUX INDEPENDANTS pour la decision de palier (2 sources d'evidence
    # differentes, meme si meme categorie) — d'ou le compte sur la liste
    # brute avant dedup, pas sur `unique_reasons`.
    signal_count = len(reasons)
    seen: set[str] = set()
    unique_reasons = [r for r in reasons if not (r in seen or seen.add(r))]

    if signal_count >= 2:
        tier = TIER_AUTO_REMOVE
    elif signal_count == 1:
        tier = TIER_LIMITED
    else:
        tier = TIER_NONE

    return {
        "tier": tier,
        "reasons": unique_reasons,
        "signal_count": signal_count,
        "signals": {
            "text": text_classification,
            "frame": frame_signals,
        },
    }
