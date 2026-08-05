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


# Taxonomie fine de motifs (2026-08quinquies, couverture max demandee),
# groupee par familles proches de ReportReason cote stream_backend
# (app/db/postgres/models/report.py). Classification en 2 PASSES (cf.
# _classify_moderation) plutot qu'un choix direct parmi ~40 motifs : ce
# modele 1.7B quantifie echoue sur des prompts a trop haute cardinalite (cf.
# _classify_category deja limite a 19 choix, et l'echec 8/8 documente dans
# classify_content pour un prompt combine plus simple que celui-ci). La
# 1ere passe choisit une FAMILLE (cles de _FAMILIES ci-dessous, <= 9 choix,
# meme ordre de grandeur que _classify_category qui fonctionne en prod) ; la
# 2e passe, seulement si une famille est detectee, affine le motif exact
# parmi les quelques valeurs de cette famille.
_FAMILIES: dict[str, list[str]] = {
    "spam": ["spam", "arnaque", "phishing", "publicite_abusive", "contenu_trompeur",
             "escroquerie_financiere", "blanchiment_argent"],
    "violence": ["violence", "menace", "incitation_violence", "gore",
                 "terrorisme", "extremisme", "armes", "fabrication_arme"],
    "haine_harcelement": ["haine", "harcelement", "cyberharcelement", "insultes",
                          "discrimination", "doxxing", "divulgation_donnees_personnelles"],
    "desinformation": ["desinformation", "manipulation", "usurpation_identite"],
    "contenu_adulte": ["nudite", "contenu_sexuel", "pornographie", "exploitation_sexuelle"],
    "danger_mineur": ["mise_en_danger_mineur"],
    "automutilation": ["suicide", "automutilation"],
    "drogue": ["drogue", "trafic_drogue"],
    "criminalite_info": ["activite_criminelle", "piratage", "malware", "vol_donnees",
                         "contenu_illegal"],
    "autre_choquant": ["contenu_choquant", "langage_vulgaire", "contenu_sensible"],
}

FLAG_REASONS = ["aucun"] + [m for motifs in _FAMILIES.values() for m in motifs]

# Reduction de la taxonomie fine ci-dessus vers les 6 ReportReason reels du
# backend -- tout motif absent de ce dict (ex. "aucun") ne cree jamais de
# Report. "desinformation"/"manipulation"/"usurpation_identite" -> misinformation
# (nouveau, jamais mappe avant 2026-08quinquies) ; le reste des categories
# choquantes/adultes/mineurs/criminalite -> inappropriate (le plus proche
# semantiquement parmi les 6 valeurs existantes, aucune categorie dediee cote
# backend pour ces cas plus graves -- cf. ReportReason, pas d'enum a ajouter
# sans migration DB).
FLAG_REASON_TO_REPORT_REASON = {
    "spam": "spam", "phishing": "spam", "publicite_abusive": "spam",
    "arnaque": "spam", "contenu_trompeur": "spam",
    "escroquerie_financiere": "spam", "blanchiment_argent": "spam",

    "violence": "violence", "menace": "violence",
    "incitation_violence": "violence", "gore": "violence",
    "terrorisme": "violence", "extremisme": "violence",
    "armes": "violence", "fabrication_arme": "violence",

    "haine": "harassment", "harcelement": "harassment",
    "cyberharcelement": "harassment", "insultes": "harassment",
    "discrimination": "harassment", "doxxing": "harassment",
    "divulgation_donnees_personnelles": "harassment",

    "desinformation": "misinformation", "manipulation": "misinformation",
    "usurpation_identite": "misinformation",

    "contenu_choquant": "inappropriate", "nudite": "inappropriate",
    "contenu_sexuel": "inappropriate", "pornographie": "inappropriate",
    "exploitation_sexuelle": "inappropriate",
    "mise_en_danger_mineur": "inappropriate", "suicide": "inappropriate",
    "automutilation": "inappropriate", "drogue": "inappropriate",
    "trafic_drogue": "inappropriate", "activite_criminelle": "inappropriate",
    "piratage": "inappropriate", "malware": "inappropriate",
    "vol_donnees": "inappropriate", "contenu_illegal": "inappropriate",
    "langage_vulgaire": "inappropriate", "contenu_sensible": "inappropriate",
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


_FAMILY_PROMPT_DESC = {
    "spam": "publicite non sollicitee, arnaque financiere, phishing, promo trompeuse",
    "violence": "menace, incitation a la violence physique, contenu extremiste ou lie aux armes",
    "haine_harcelement": "discours haineux/discriminatoire, harcelement, insultes, doxxing",
    "desinformation": "le texte affirme que TOUTES les informations officielles sont des "
                       "mensonges, incite a partager en urgence 'avant suppression', ou "
                       "appelle explicitement au boycott general d'une institution sur cette "
                       "base. Une simple critique ou opinion politique, meme severe, N'EN "
                       "EST PAS",
    "contenu_adulte": "nudite, contenu sexuel explicite",
    "danger_mineur": "mise en danger ou exploitation d'un mineur",
    "automutilation": "incitation au suicide ou a l'automutilation",
    "drogue": "vente ou promotion de drogue illegale",
    "criminalite_info": "activite criminelle organisee, piratage informatique, virus",
    "autre_choquant": "contenu choquant/violent visuellement decrit, langage tres vulgaire",
}


def _classify_family(llm, content: str) -> str | None:
    """1ere passe (2026-08quinquies) : choisit une FAMILLE de risque parmi 10,
    pas un motif fin directement -- cardinalite comparable a
    _classify_category (19 choix, fonctionne en prod), contrairement a un
    choix direct parmi les ~40 motifs fins de FLAG_REASONS qui depasserait la
    capacite fiable de ce modele 1.7B quantifie (cf. classify_content pour
    l'echec 8/8 deja constate sur un prompt combine plus simple que celui-ci).

    Volontairement calibre sur des cas nets, pas sur l'ambigu : cf.
    _classify_reason pour le detail du compromis faux-negatifs > faux-positifs
    deja adopte pour haine/harcelement (tentative d'elargissement revertee le
    2026-07-31, ce petit modele ne distingue pas le sens figure).
    """
    families = "\n".join(f"- {name} : {desc}" for name, desc in _FAMILY_PROMPT_DESC.items())
    prompt = (
        "Ce contenu correspond-il a une des categories de risque suivantes ?\n"
        f"{families}\n"
        "Si aucune ne correspond clairement, reponds 'aucun'. En cas de doute, "
        "reponds 'aucun' plutot que de deviner.\n"
        f"Reponds avec un seul mot parmi: {', '.join(_FAMILIES.keys())}, aucun.\n"
        f"Contenu: {content} /no_think"
    )
    family = _chat(llm, prompt, max_tokens=10)
    return family if family in _FAMILIES else None


def _classify_reason(llm, content: str, family: str) -> str:
    """2e passe, appelee seulement si _classify_family a detecte une famille --
    affine le motif exact parmi les quelques valeurs de cette famille (3-8
    choix), cardinalite faible donc fiable meme sur ce petit modele."""
    motifs = _FAMILIES[family]
    if len(motifs) == 1:
        return motifs[0]
    prompt = (
        f"Ce contenu a ete identifie comme relevant de la categorie '{family}'. "
        f"Quel motif precis parmi: {', '.join(motifs)} decrit le mieux ce contenu ?\n"
        "Reponds seulement avec le mot du motif, rien d'autre.\n"
        f"Contenu: {content} /no_think"
    )
    reason = _chat(llm, prompt, max_tokens=10)
    return reason if reason in motifs else motifs[0]


def _classify_moderation(llm, content: str) -> str:
    family = _classify_family(llm, content)
    if family is None:
        return "aucun"
    return _classify_reason(llm, content, family)


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
    testes. La moderation elle-meme est en 2 passes depuis 2026-08quinquies
    (cf. _classify_moderation) pour couvrir ~40 motifs fins sans depasser la
    cardinalite fiable du modele : 2-3 appels LLM au total selon qu'une
    famille de risque est detectee ou non. Plus lent que l'ancien schema a 2
    appels fixes, mais c'est le signal de moderation qui compte le plus ici,
    pas la latence.

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
