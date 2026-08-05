"""Embeddings semantiques d'image via OpenCLIP (ViT-B-32, cf. ARCHITECTURE_IA.md section 3).

Utilise pour la recherche par similarite (pgvector), comme feature d'appoint
pour text_pipeline.classify_content, et comme base du classifieur NSFW
(moderation_pipeline.py) — la detection NSFW reutilise directement l'embedding
CLIP deja calcule (zero-shot par similarite texte/image), pas de second reseau
charge en RAM (cf. ARCHITECTURE_IA.md section 9 pour la justification du choix).
"""

import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.model_manager import model_manager

# Prompts zero-shot pour la classification NSFW par similarite CLIP (approche
# "CLIP-based classifier" — cf. https://github.com/LAION-AI/CLIP-based-NSFW-Detector
# et papers zero-shot CLIP : chaque prompt est encode une fois avec le meme
# modele CLIP que les images, la classe retenue est celle dont le texte est le
# plus proche cosinus de l'embedding image. Pas de poids supplementaires a
# telecharger/charger : seul CLIP (deja en memoire) est utilise.
_NSFW_PROMPT_GROUPS = {
    "sexual_explicit": [
        "a photo of explicit sexual content",
        "a photo of nudity, naked body",
        "a pornographic image",
    ],
    "suggestive": [
        "a photo of a person in lingerie or swimwear, suggestive pose",
    ],
    "safe": [
        "a photo of people fully clothed",
        "a photo of everyday life, objects, or scenery",
        "a safe for work photo",
    ],
}

# Ordre fige des groupes, utilise pour aligner les scores softmax sur les cles
_NSFW_LABELS = list(_NSFW_PROMPT_GROUPS.keys())
_NSFW_PROMPTS = [p for group in _NSFW_PROMPT_GROUPS.values() for p in group]
_NSFW_GROUP_SIZES = [len(group) for group in _NSFW_PROMPT_GROUPS.values()]


def _load_clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model_name, pretrained=settings.clip_pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
    return {"model": model, "preprocess": preprocess, "tokenizer": tokenizer}


def _clip_bundle():
    return model_manager.get("clip", _load_clip)


def _image_features(image_path: str):
    import torch
    from PIL import Image

    bundle = _clip_bundle()
    model, preprocess = bundle["model"], bundle["preprocess"]

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def download_image(image_url: str) -> str | None:
    """Telecharge une image distante (R2/S3, cf. post/event/concert qui ne
    fournissent qu'une URL, contrairement au reel dont video_pipeline extrait
    deja des frames locales via ffmpeg) vers un fichier temporaire local.

    Retourne None en cas d'echec (URL invalide, timeout, contenu non-image)
    plutot que de lever — meme principe de degradation gracieuse que
    video_pipeline.extract_keyframes (un post/event/concert doit rester
    analysable meme si le telechargement de son image echoue, avec
    simplement moins de signaux disponibles pour le verdict).

    User-Agent explicite requis : constate en test (2026-08-05) que des CDN
    publics (Wikimedia) renvoient 403 Forbidden aux requetes sans User-Agent
    identifiable -- httpx n'en envoie aucun par defaut. R2/S3 (stockage reel
    des images post/event/concert) ne bloque pas sans User-Agent, mais un
    header explicite est sans risque et evite la meme classe de faux negatif
    silencieux si un CDN devant R2/S3 applique un jour une regle similaire."""
    import httpx

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; GoFolyXModerationBot/1.0)"}
        with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
            resp = client.get(image_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None
            suffix = Path(image_url.split("?", 1)[0]).suffix or ".jpg"
            out_path = str(Path(tempfile.mkdtemp(prefix="ai_image_")) / f"image{suffix}")
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
    except Exception:
        return None


def embed_image(image_path: str) -> list[float]:
    """Retourne un vecteur d'embedding normalise (dimension du modele CLIP, ex. 512)."""
    return _image_features(image_path).squeeze(0).tolist()


def classify_nsfw(image_path: str) -> dict:
    """Classifie une image en {"label": str, "confidence": float, "scores": dict,
    "margin": float}.

    `label` est un des groupes de `_NSFW_LABELS` ("sexual_explicit", "suggestive",
    "safe"). Zero-shot CLIP : encode les prompts et l'image dans le meme espace.

    Decision basee sur la MARGE de similarite cosinus BRUTE (groupe gagnant -
    groupe "safe"), pas sur une probabilite softmax — corrige un vrai probleme
    de calibration constate 2026-08-05 : la temperature *20 (choisie pour eviter
    un faux positif sur une photo de couteau de cuisine, cf. historique ci-dessous)
    ecrasait aussi les VRAIS positifs (contenu explicite reel jamais detecte,
    confidence n'atteignant jamais le seuil 0.8 quel que soit le contenu) — mesure
    texte-texte confirmant un chevauchement fort entre prompts "explicit" et
    "safe" (jusqu'a 0.655 de similarite croisee), rendant tout seuil absolu sur
    une probabilite softmax fragile dans un sens ou l'autre selon la temperature.
    Une marge sur les scores cosinus bruts (avant toute distortion exponentielle)
    est plus stable : `margin` de quelques centiemes suffit deja a distinguer les
    groupes de maniere fiable, sans qu'un choix de temperature n'ecrase le signal.

    Historique temperature softmax (gardee ci-dessous pour `scores`/`confidence`,
    desormais informatifs seulement, la decision reelle utilise `margin`) :
    mesure reelle 2026-07-30 montrait des similarites cosinus brutes tassees
    entre 0.01 et 0.18 — avec *100, un ecart de 0.02 (bruit) devenait un facteur
    e^2~7x en probabilite (faux positif 60% "sexual explicit" sur un couteau de
    cuisine). *20 attenue cet effet mais reste un palliatif : CLIP ViT-B-32 en
    zero-shot a un pouvoir discriminant limite sur ce cas precis (cf.
    ARCHITECTURE_IA.md section 9).
    """
    import torch

    bundle = _clip_bundle()
    model, tokenizer = bundle["model"], bundle["tokenizer"]

    image_features = _image_features(image_path)

    with torch.no_grad():
        text_tokens = tokenizer(_NSFW_PROMPTS)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0)

        # Moyenne des prompts par groupe, pour ne pas favoriser un groupe
        # uniquement parce qu'il a plus de formulations de prompt.
        group_scores = []
        offset = 0
        for size in _NSFW_GROUP_SIZES:
            group_scores.append(similarities[offset:offset + size].mean())
            offset += size
        group_scores = torch.stack(group_scores)

        probs = torch.softmax(group_scores * 20.0, dim=0)

    raw_scores = {label: float(s) for label, s in zip(_NSFW_LABELS, group_scores)}
    safe_score = raw_scores["safe"]
    # Marge = score brut du groupe le plus eleve HORS "safe" moins le score
    # "safe" lui-meme -- positif => l'image ressemble davantage a ce groupe
    # qu'a du contenu sur (le vrai signal de decision, cf. docstring).
    non_safe_best_label = max(
        (l for l in _NSFW_LABELS if l != "safe"), key=lambda l: raw_scores[l],
    )
    margin = raw_scores[non_safe_best_label] - safe_score

    # `label`/`confidence` derives de la marge (decision reelle), pas du softmax
    # (garde uniquement a titre informatif dans `scores`, cf. docstring).
    label = non_safe_best_label if margin > 0 else "safe"
    scores = {l: round(float(p), 4) for l, p in zip(_NSFW_LABELS, probs)}

    return {"label": label, "confidence": scores[label], "scores": scores, "margin": round(margin, 4)}
