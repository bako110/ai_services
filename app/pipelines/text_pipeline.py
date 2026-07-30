"""Tagging de categorie et moderation legere via un LLM quantifie (Qwen3 GGUF).

Voir ARCHITECTURE.md section 3 pour la taxonomie fermee de categories —
c'est la seule liste de sortie autorisee pour `category`, le prompt doit la
forcer explicitement pour eviter que le modele invente ses propres libelles.
"""

from app.core.config import settings
from app.core.model_manager import model_manager

CATEGORIES = [
    "musique", "sport", "gaming", "humour", "danse", "cuisine", "mode",
    "beaute", "tech", "education", "lifestyle", "art", "voyage", "business",
    "actualite", "spiritualite", "famille", "sante", "autre",
]


def _load_llm():
    from llama_cpp import Llama

    return Llama(model_path=settings.text_model_path, n_ctx=2048, n_threads=4)


# Motifs de signalement que le LLM peut choisir — alignes sur ReportReason
# cote stream_backend (app/db/postgres/models/report.py) pour que
# moderation_pipeline.py puisse creer un Report avec la meme taxonomie sans
# table de correspondance separee.
FLAG_REASONS = ["spam", "arnaque", "haine", "violence", "aucun"]


def classify_content(
    title: str = "",
    description: str = "",
    transcription: str = "",
    detected_objects: list[str] | None = None,
) -> dict:
    """Retourne {"category": str, "confidence": float, "flagged": bool, "flag_reason": str|None}.

    `confidence` basse (< 0.5) => appelant laisse `category` a null en base
    plutot que d'ecrire une valeur peu fiable (cf. ARCHITECTURE_IA.md section 5).

    `flagged`/`flag_reason` : signal de moderation textuelle (spam, arnaque,
    propos haineux, incitation a la violence) — combine avec les signaux
    image/objets dans moderation_pipeline.py, jamais utilise seul pour agir.
    """
    llm = model_manager.get("text_llm", _load_llm)

    context_parts = [p for p in [title, description, transcription] if p]
    if detected_objects:
        context_parts.append("Objets detectes: " + ", ".join(detected_objects))
    context = "\n".join(context_parts).strip()

    if not context:
        return {"category": None, "confidence": 0.0, "flagged": False, "flag_reason": None}

    prompt = (
        "Tu classes et moderes du contenu pour une app de streaming. "
        f"Categories autorisees (choisis-en une seule, exactement): {', '.join(CATEGORIES)}.\n"
        f"Motifs de signalement autorises (choisis-en un seul, exactement): {', '.join(FLAG_REASONS)}.\n"
        "'arnaque' = promesse de gain irrealiste, fausse promo, lien suspect, "
        "demande d'argent/coordonnees bancaires deguisee.\n"
        f"Contenu:\n{context}\n\n"
        "Reponds UNIQUEMENT avec les 3 valeurs choisies separees par | "
        "(pas de texte, pas les noms des champs). "
        "Exemple de format exact attendu: musique|0.9|aucun /no_think"
    )
    # create_chat_completion (pas l'appel completion brut `llm(prompt, ...)`)
    # est necessaire : Qwen3-Instruct est entraine sur un chat template
    # (<|im_start|>user...<|im_end|>), un prompt brut le fait "continuer le
    # texte" au lieu d'y repondre — constate en prod le 2026-07-30, le modele
    # completait litteralement "confiance(0-1)" par du texte d'exemple au lieu
    # de classifier. Le suffixe "/no_think" desactive le mode raisonnement de
    # Qwen3 (sinon le budget max_tokens est consomme par un bloc <think>...</think>
    # avant d'atteindre la reponse).
    raw = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.0,
    )["choices"][0]["message"]["content"].strip()

    return _parse_response(raw)


def _parse_response(raw: str) -> dict:
    # Qwen3 emet toujours un bloc <think>...</think> avant la reponse, meme
    # vide avec /no_think ("<think>\n\n</think>\n\ncategorie|conf|motif") —
    # ne garder que ce qui suit la derniere balise fermante.
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1].strip()
    try:
        category, confidence, flag_reason = [p.strip().lower() for p in raw.split("|")]
        if category not in CATEGORIES:
            category = "autre"
        if flag_reason not in FLAG_REASONS:
            flag_reason = "aucun"
        flagged = flag_reason != "aucun"
        return {
            "category": category,
            "confidence": float(confidence),
            "flagged": flagged,
            "flag_reason": flag_reason if flagged else None,
        }
    except (ValueError, IndexError):
        return {"category": None, "confidence": 0.0, "flagged": False, "flag_reason": None}
