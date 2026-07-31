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


def _chat(llm, prompt: str, max_tokens: int) -> str:
    """create_chat_completion (pas l'appel completion brut `llm(prompt, ...)`)
    est necessaire : Qwen3-Instruct est entraine sur un chat template
    (<|im_start|>user...<|im_end|>), un prompt brut le fait "continuer le
    texte" au lieu d'y repondre — constate en prod le 2026-07-30, le modele
    completait litteralement "confiance(0-1)" par du texte d'exemple au lieu
    de classifier. Le suffixe "/no_think" desactive le mode raisonnement de
    Qwen3 (sinon le budget max_tokens est consomme par un bloc
    <think>...</think> avant d'atteindre la reponse).
    """
    raw = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )["choices"][0]["message"]["content"]
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1]
    return raw.strip().lower()


def _classify_category(llm, content: str) -> str:
    prompt = (
        "Choisis UNE SEULE categorie dans cette liste qui decrit le mieux ce "
        f"contenu: {', '.join(CATEGORIES)}.\n"
        f"Contenu: {content}\n"
        "Reponds seulement avec le mot de la categorie, rien d'autre. /no_think"
    )
    category = _chat(llm, prompt, max_tokens=10)
    return category if category in CATEGORIES else "autre"


def _classify_moderation(llm, content: str) -> str:
    prompt = (
        "Ce contenu contient-il une arnaque, un discours haineux, une "
        "incitation a la violence, ou du spam ? Une arnaque est une promesse "
        "de gain irrealiste, une fausse promo, ou une demande d'argent/"
        "coordonnees bancaires deguisee. Reponds avec un seul mot parmi: "
        "arnaque, haine, violence, spam, aucun.\n"
        f"Contenu: {content} /no_think"
    )
    flag_reason = _chat(llm, prompt, max_tokens=10)
    return flag_reason if flag_reason in FLAG_REASONS else "aucun"


def classify_content(
    title: str = "",
    description: str = "",
    transcription: str = "",
    detected_objects: list[str] | None = None,
) -> dict:
    """Retourne {"category": str, "confidence": float, "flagged": bool, "flag_reason": str|None}.

    Deux appels LLM separes (categorie, puis moderation) plutot qu'un seul
    prompt combine — constate en prod le 2026-07-30 : un prompt unique
    demandant categorie+motif+format pipe strict etait trop complexe pour
    Qwen3-1.7B quantifie, qui repondait au bon format mais avec un contenu
    faux (ex. "GAGNEZ 5000 EUROS... envoyez vos coordonnees bancaires"
    classe "gaming|0.9|aucun", aucun cas de test detecte sur 8/8 essais).
    Decouple, chaque prompt simple et isole, la moderation seule a detecte
    4/4 cas de test (arnaque/haine/violence/spam) sans faux positif sur les
    cas neutres testes. Plus lent (2x le cout d'inference), mais c'est le
    signal de moderation qui compte le plus ici, pas la latence.

    `confidence` n'est plus produite par le LLM (le modele ne l'estimait pas
    de facon fiable) — fixee a 1.0 si une categorie valide est retournee,
    cf. ARCHITECTURE_IA.md section 5 pour la logique cote appelant (confiance
    basse => `category` laissee a null en base).

    `flagged`/`flag_reason` : signal de moderation textuelle — combine avec
    les signaux image/objets dans moderation_pipeline.py, jamais utilise seul
    pour agir (jamais de suppression automatique).
    """
    llm = model_manager.get("text_llm", _load_llm)

    context_parts = [p for p in [title, description, transcription] if p]
    if detected_objects:
        context_parts.append("Objets detectes: " + ", ".join(detected_objects))
    context = "\n".join(context_parts).strip()

    if not context:
        return {"category": None, "confidence": 0.0, "flagged": False, "flag_reason": None}

    category = _classify_category(llm, context)
    flag_reason = _classify_moderation(llm, context)
    flagged = flag_reason != "aucun"

    return {
        "category": category,
        "confidence": 1.0,
        "flagged": flagged,
        "flag_reason": flag_reason if flagged else None,
    }
