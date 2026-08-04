"""Stockage et recherche d'embeddings via pgvector, sur le Postgres deja utilise
par stream_backend (pas de service vector DB separe — cf. ARCHITECTURE_IA.md section 3).

Migration attendue sur la base existante (a ecrire au moment de coder, cf.
ARCHITECTURE_IA.md section 8) :
    CREATE EXTENSION IF NOT EXISTS vector;
    ALTER TABLE reels ADD COLUMN embedding vector(512);
    CREATE INDEX ON reels USING ivfflat (embedding vector_cosine_ops);

Tables reels/posts/lives verifiees dans stream_backend/app/db/postgres/models/ —
noms de table corrects (reels, posts, lives), toutes ont deja une colonne
`category` (String(30) nullable), pas encore de colonne `embedding`.

DATABASE_URL ici doit etre un DSN direct asyncpg (postgresql://...), PAS le
format SQLAlchemy `postgresql+asyncpg://...?prepared_statement_cache_size=0`
utilise par stream_backend/app/config.py — asyncpg.connect() ne comprend ni
le prefixe `+asyncpg` ni ce query param.
"""

import asyncpg
from pgvector.asyncpg import register_vector

from app.core.config import settings


async def _connect() -> asyncpg.Connection:
    """asyncpg brut ne sait pas encoder une list[float] Python vers le type
    `vector` Postgres (constate en prod 2026-08 : DataError "expected str,
    got list" au premier vrai appel de save_embedding contre la colonne
    reelle -- jamais declenche avant car la colonne n'existait pas encore
    lors des tests precedents). register_vector() enregistre le codec
    pgvector sur CETTE connexion — necessaire avant toute requete qui lit ou
    ecrit une colonne `vector`."""
    conn = await asyncpg.connect(settings.database_url)
    await register_vector(conn)
    return conn


async def save_embedding(content_type: str, content_id: str, embedding: list[float]) -> None:
    table = {"reel": "reels", "post": "posts", "live": "lives"}[content_type]
    conn = await _connect()
    try:
        await conn.execute(
            f"UPDATE {table} SET embedding = $1 WHERE id = $2",
            embedding, content_id,
        )
    finally:
        await conn.close()


# normalize_category(None) (stream_backend/app/utils/content_category.py)
# retourne toujours "autre", jamais NULL -- toute ligne creee via ReelService.
# create_reel a donc systematiquement category="autre" en absence de choix du
# createur, jamais NULL. Un filtre `WHERE category IS NULL` ne matcherait
# donc jamais rien (constate 2026-07-31 avant meme le premier cablage reel).
DEFAULT_CATEGORY = "autre"


async def save_category(content_type: str, content_id: str, category: str) -> None:
    """Ecrit la categorie inferee dans la colonne `category` existante
    (deja presente sur reels/posts/lives/communities, cf. reel.py etc.).

    Ne remplace que `category = "autre"`. Limite connue : impossible de
    distinguer en base un "autre" par defaut (createur n'a rien choisi) d'un
    "autre" choisi expres par le createur — les deux cas sont ecrases par
    l'IA de la meme facon. Compromis accepte faute d'une colonne separee
    (ex. category_is_explicit) qui n'existe pas dans le schema actuel."""
    table = {"reel": "reels", "post": "posts", "live": "lives"}[content_type]
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(
            f"UPDATE {table} SET category = $1 WHERE id = $2 AND category = $3",
            category, content_id, DEFAULT_CATEGORY,
        )
    finally:
        await conn.close()


# Mapping palier de moderation -> statut reel (stream_backend/app/db/postgres/
# models/reel.py::ReelStatus). "auto_remove" -> archived (pas de suppression
# definitive, cf. block_reported_content existant qui fait la meme chose pour
# les signalements humains -- reversible par un admin, contrairement a
# delete_reported_content). "limited" -> nouveau statut ReelStatus.limited,
# ajoute 2026-08 (ALTER TYPE reelstatus ADD VALUE 'limited').
_TIER_TO_REEL_STATUS = {"auto_remove": "archived", "limited": "limited"}

# Reason par defaut si `reasons` est vide malgre un tier != "none" -- ne
# devrait jamais arriver en pratique (build_verdict ne renvoie un tier positif
# que s'il y a au moins 1 reason), mais Report.reason est NOT NULL cote
# backend donc il faut une valeur de repli plutot que de planter l'ecriture.
_FALLBACK_REPORT_REASON = "other"


async def apply_moderation_verdict(
    content_type: str,
    content_id: str,
    verdict: dict,
    reporter_id: str,
) -> None:
    """Applique le verdict de moderation_pipeline.build_verdict() en base :
    cree un Report (meme table que les signalements humains, reporter_id =
    compte systeme dedie) et, pour les paliers "limited"/"auto_remove",
    change le statut du contenu. Ne fait rien si tier == "none".

    Jamais de suppression definitive (DELETE) : "auto_remove" passe le
    contenu a status="archived", reversible par un admin via l'interface de
    moderation existante -- coherent avec le principe deja pose de ne jamais
    laisser l'IA prendre une decision irreversible seule.
    """
    tier = verdict.get("tier", "none")
    if tier == "none":
        return

    reasons = verdict.get("reasons") or [_FALLBACK_REPORT_REASON]
    table = {"reel": "reels", "post": "posts", "live": "lives"}[content_type]
    new_status = _TIER_TO_REEL_STATUS.get(tier)

    conn = await asyncpg.connect(settings.database_url)
    try:
        async with conn.transaction():
            # Pas de contrainte unique en base sur (reporter_id, content_type,
            # content_id) — seule create_report() la fait cote applicatif pour
            # les signalements humains. Verification manuelle ici pour eviter
            # des doublons si analyze_reel est relance (retry Celery, re-run
            # manuel) sur un contenu deja modere par ce meme compte systeme.
            already_reported = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM reports WHERE reporter_id = $1 AND content_type = $2 AND content_id = $3)",
                reporter_id, content_type, content_id,
            )
            if already_reported:
                return

            for reason in reasons:
                await conn.execute(
                    """
                    INSERT INTO reports (id, reporter_id, content_type, content_id, reason, details, status, created_at)
                    VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, now())
                    """,
                    reporter_id, content_type, content_id, reason,
                    f"Signale automatiquement par ai_service (palier: {tier})",
                    "resolved" if tier == "auto_remove" else "pending",
                )

            if new_status and table == "reels":
                await conn.execute(
                    f"UPDATE {table} SET status = $1 WHERE id = $2 AND status = 'published'",
                    new_status, content_id,
                )
    finally:
        await conn.close()


async def find_similar(content_type: str, embedding: list[float], limit: int = 10) -> list[dict]:
    table = {"reel": "reels", "post": "posts", "live": "lives"}[content_type]
    conn = await _connect()
    try:
        rows = await conn.fetch(
            f"SELECT id, embedding <=> $1 AS distance FROM {table} "
            f"WHERE embedding IS NOT NULL ORDER BY distance ASC LIMIT $2",
            embedding, limit,
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()
