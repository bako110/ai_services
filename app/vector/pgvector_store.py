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

from app.core.config import settings


async def save_embedding(content_type: str, content_id: str, embedding: list[float]) -> None:
    table = {"reel": "reels", "post": "posts", "live": "lives"}[content_type]
    conn = await asyncpg.connect(settings.database_url)
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


async def find_similar(content_type: str, embedding: list[float], limit: int = 10) -> list[dict]:
    table = {"reel": "reels", "post": "posts", "live": "lives"}[content_type]
    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch(
            f"SELECT id, embedding <=> $1 AS distance FROM {table} "
            f"WHERE embedding IS NOT NULL ORDER BY distance ASC LIMIT $2",
            embedding, limit,
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()
