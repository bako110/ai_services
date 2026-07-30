"""Embeddings semantiques d'image via OpenCLIP (ViT-B-32, cf. ARCHITECTURE_IA.md section 3).

Utilise pour la recherche par similarite (pgvector), comme feature d'appoint
pour text_pipeline.classify_content, et comme base du classifieur NSFW
(moderation_pipeline.py) — la detection NSFW reutilise directement l'embedding
CLIP deja calcule (zero-shot par similarite texte/image), pas de second reseau
charge en RAM (cf. ARCHITECTURE_IA.md section 9 pour la justification du choix).
"""

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


def embed_image(image_path: str) -> list[float]:
    """Retourne un vecteur d'embedding normalise (dimension du modele CLIP, ex. 512)."""
    return _image_features(image_path).squeeze(0).tolist()


def classify_nsfw(image_path: str) -> dict:
    """Classifie une image en {"label": str, "confidence": float, "scores": dict}.

    `label` est un des groupes de `_NSFW_LABELS` ("sexual_explicit", "suggestive",
    "safe"). Zero-shot CLIP : encode les prompts et l'image dans le meme espace,
    softmax sur les similarites cosinus. Pas de seuil applique ici — c'est
    moderation_pipeline.py qui decide de l'action a partir de `label`/`confidence`.
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

        # Moyenne des prompts par groupe avant softmax, pour ne pas favoriser
        # un groupe uniquement parce qu'il a plus de formulations de prompt.
        group_scores = []
        offset = 0
        for size in _NSFW_GROUP_SIZES:
            group_scores.append(similarities[offset:offset + size].mean())
            offset += size
        group_scores = torch.stack(group_scores)

        probs = torch.softmax(group_scores * 100.0, dim=0)  # temperature CLIP standard

    scores = {label: round(float(p), 4) for label, p in zip(_NSFW_LABELS, probs)}
    best_label = max(scores, key=scores.get)

    return {"label": best_label, "confidence": scores[best_label], "scores": scores}
