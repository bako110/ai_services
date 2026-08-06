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
#
# Historique de calibration (2026-08quinquies/sexies) : une tentative
# d'elargissement a ~40 motifs fins groupes en 10 familles a ete testee en
# prod le 2026-08-06 et ABANDONNEE -- plusieurs formats de prompt essayes
# (choix unique parmi 10 familles, choix binaire sequentiel par famille,
# choix unique a 5-6 motifs combinant desinformation avec les 4 motifs
# d'origine) ont tous echoue sur un jeu de 6 textes de test simples (texte
# neutre, critique politique legitime, arnaque, violence, haine, spam) :
# soit repli excessif sur "aucun" par noyade sur la cardinalite, soit biais
# d'acquiescement massif (un texte neutre repondait "oui" a la moitie des
# familles testees). Seul le motif "desinformation", pose en QUESTION
# BINAIRE ISOLEE (jamais combinee dans le meme prompt que les 4 motifs
# d'origine, cf. _classify_desinformation) avec le format "conclusion
# d'abord puis justification courte", s'est montre fiable sur tous les cas
# testes ce jour-la. D'ou ce choix : garder le prompt _classify_moderation
# EXACTEMENT tel qu'a l'origine (ne pas risquer de le degrader plus avant
# sans nouveau test complet), et ajouter desinformation comme verification
# supplementaire separee, seulement si aucun des 4 motifs d'origine n'a
# matche. Mieux vaut une couverture etroite mais fiable qu'une couverture
# large qui genere des faux positifs massifs.
FLAG_REASONS = ["spam", "arnaque", "haine", "violence", "desinformation", "aucun"]

FLAG_REASON_TO_REPORT_REASON = {
    "spam": "spam",
    "arnaque": "spam",
    "haine": "harassment",
    "violence": "violence",
    "desinformation": "misinformation",
}


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
    """Volontairement calibre sur des cas nets (arnaque/haine/violence/spam
    "graves"), pas sur les insultes legeres ou ambigues. Tentative
    d'elargissement testee le 2026-07-31 : demander au modele de detecter
    aussi les insultes courtes type "tu es malade" faisait remonter le taux
    de detection de ce cas precis, mais generait de nouveaux faux positifs
    sur des phrases positives ambigues ("tu es malade ce truc est stylé",
    expression figuree) — ce petit modele 1.7B ne distingue pas le sens
    figure sans plus de contexte. Revert : mieux vaut louper une insulte
    legere que flaguer a tort un message bienveillant, cf. principe deja
    pose de minimiser les faux positifs tant qu'aucune revue humaine
    n'absorbe le signal en amont. Les insultes/harcelement plus subtils
    restent geres par le signalement utilisateur existant
    (stream_backend/app/services/report_service.py), pas par ce signal IA.

    "desinformation" (2026-08sexies) verifiee dans un APPEL LLM SEPARE, pas
    ajoutee comme 5e choix dans ce meme prompt -- teste en prod le
    2026-08-06 : l'ajouter comme choix supplementaire ici degradait aussi la
    detection des 4 motifs d'origine (2/6 cas corrects seulement sur un jeu
    de 6 textes de test, y compris des ratés sur des cas nets de haine/
    violence/spam que le prompt a 4 choix seul gere mieux). Cf.
    _classify_desinformation pour le detail de cette 2e question, jamais
    combinee dans le meme appel.
    """
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


def _classify_desinformation(llm, content: str) -> bool:
    """Question binaire ISOLEE (jamais combinee avec _classify_moderation
    dans le meme prompt, cf. sa docstring) -- cible explicitement le PATTERN
    manipulatoire (rejet en bloc de toute source officielle comme
    "mensonge", appel a partager en urgence/avant suppression, incitation au
    boycott general), jamais la simple opinion/critique politique.

    Format "conclusion d'abord + justification courte" valide en test reel
    le 2026-08-06 : seul format/motif reste fiable sur 3/3 cas testes (texte
    neutre -> NON, critique politique legitime -> NON, cas net avec les 3
    marqueurs presents -> OUI avec justification coherente) parmi 6 formats
    de prompt differents testes ce jour-la sur ce modele 1.7B quantifie.
    """
    prompt = (
        "Question: Ce texte affirme-t-il explicitement que TOUTES les "
        "informations officielles sont des mensonges, ET incite-t-il "
        "explicitement a partager en urgence avant suppression, ET "
        "appelle-t-il explicitement a un boycott general d'une institution "
        "sur cette base ? Une simple critique ou opinion politique, meme "
        "severe, ne compte PAS.\n"
        f"Texte a analyser: \"{content}\"\n"
        "Reponds d'abord par UN SEUL MOT sur la premiere ligne: OUI ou NON. "
        "Puis explique brievement sur la ligne suivante. /no_think"
    )
    raw = _chat(llm, prompt, max_tokens=60)
    return raw.split("\n")[0].strip().startswith("oui")


def classify_content(
    title: str = "",
    description: str = "",
    transcription: str = "",
    detected_objects: list[str] | None = None,
) -> dict:
    """Retourne {"category": str, "confidence": float, "flagged": bool, "flag_reason": str|None}.

    Appels LLM separes (categorie, puis moderation) plutot qu'un seul prompt
    combine — constate en prod le 2026-07-30 : un prompt unique demandant
    categorie+motif+format pipe strict etait trop complexe pour Qwen3-1.7B
    quantifie, qui repondait au bon format mais avec un contenu faux (ex.
    "GAGNEZ 5000 EUROS... envoyez vos coordonnees bancaires" classe
    "gaming|0.9|aucun", aucun cas de test detecte sur 8/8 essais). Decouple,
    chaque prompt simple et isole, la moderation seule a detecte 4/4 cas de
    test (arnaque/haine/violence/spam) sans faux positif sur les cas neutres
    testes. 2-3 appels LLM au total depuis 2026-08sexies : categorie,
    moderation classique, puis desinformation seulement si rien d'autre n'a
    matche (cf. _classify_desinformation) -- plus lent, mais c'est le signal
    de moderation qui compte le plus ici, pas la latence.

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
    if flag_reason == "aucun" and _classify_desinformation(llm, context):
        flag_reason = "desinformation"
    flagged = flag_reason != "aucun"

    return {
        "category": category,
        "confidence": 1.0,
        "flagged": flagged,
        "flag_reason": flag_reason if flagged else None,
    }
