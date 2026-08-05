"""Stockage et recherche d'embeddings via pgvector, sur le Postgres deja utilise
par stream_backend (pas de service vector DB separe — cf. ARCHITECTURE_IA.md section 3).

DATABASE_URL ici doit etre un DSN direct asyncpg (postgresql://...), PAS le
format SQLAlchemy `postgresql+asyncpg://...?prepared_statement_cache_size=0`
utilise par stream_backend/app/config.py — asyncpg.connect() ne comprend ni
le prefixe `+asyncpg` ni ce query param.

Extension posts/events/concerts (2026-08) : le mapping table est desormais
partage par toutes les fonctions via _TABLE_FOR_TYPE, pour eviter que
certains content_type soient supportes par certaines fonctions mais pas
d'autres (constate : apply_moderation_verdict ne changeait le statut QUE
pour les reels avant cette refonte, alors que le mapping table existait deja
pour post/live -- bug latent jamais declenche faute d'appelant).
"""

import asyncpg
import redis.asyncio as aioredis
from pgvector.asyncpg import register_vector

from app.core.config import settings

# Prefixes de cache Redis (cote stream_backend, app/utils/cache.py) a
# invalider quand un changement de statut IA modifie un contenu deja publie.
# Bug corrige (2026-08) : apply_moderation_verdict changeait bien `status` en
# base (via cette connexion asyncpg directe, hors du backend HTTP), mais ne
# passait jamais par cache_invalidate_prefix -- le feed public restait servi
# depuis le cache (TTL 300s, cf. posts.py feed()) jusqu'a 5 min APRES qu'un
# contenu detecte comme inapproprie soit passe a status="limited", le
# laissant visible de tous entre-temps malgre la moderation reelle en base.
_CACHE_PREFIXES_BY_TYPE = {
    "reel":    ["reels:list:", "reels:user:", "reel:"],
    "post":    ["posts:feed:", "posts:user:", "post:"],
    "event":   ["events:list:", "event:"],
    "concert": ["concerts:list:", "concerts:"],
}


def _backend_cache_redis_url() -> str:
    """settings.redis_url (ai_service) pointe sur la base logique Redis /2,
    dediee au result backend Celery -- DIFFERENTE de la base /0 utilisee par
    le cache applicatif cote stream_backend (app/utils/cache.py). Bug
    constate 2026-08-05 : l'invalidation de cache ecrivait bien dans Redis,
    mais dans la base /2, donc totalement invisible du cache reel /0 -- le
    feed public restait servi en cache malgre le changement de statut. Ce
    helper force la base /0 en remplacant le suffixe, sans dupliquer une
    URL complete en config (evite un redeploiement de .env supplementaire)."""
    url = settings.redis_url
    if "/" in url.rsplit("@", 1)[-1]:
        base, _, _ = url.rpartition("/")
        return f"{base}/0"
    return f"{url}/0"


async def _invalidate_content_cache(content_type: str, content_id: str) -> None:
    """Best-effort : n'echoue jamais bruyamment si Redis est indisponible ou
    l'URL pointe vers localhost (meme garde-fou que cache.py cote backend)."""
    if "localhost" in settings.redis_url or "127.0.0.1" in settings.redis_url:
        return
    try:
        r = aioredis.from_url(_backend_cache_redis_url(), encoding="utf-8", decode_responses=True)
        try:
            for prefix in _CACHE_PREFIXES_BY_TYPE.get(content_type, []):
                cursor = 0
                to_delete: list[str] = []
                while True:
                    cursor, keys = await r.scan(cursor, match=f"{prefix}*", count=100)
                    to_delete.extend(keys)
                    if cursor == 0:
                        break
                if to_delete:
                    await r.delete(*to_delete)
            # Cle exacte du detail (post:{id}, reel:{id}, event:{id},
            # concerts:{id}) -- deja couverte par le prefixe ci-dessus dans
            # 3 cas sur 4, sauf "concerts:{id}" qui partage son prefixe avec
            # "concerts:list:" (meme prefixe "concerts:"), donc redondant
            # mais explicite volontairement pour ne pas dependre d'un detail
            # de nommage fragile.
        finally:
            await r.aclose()
    except Exception:
        pass

# Mapping content_type (utilise cote ai_service, taches Celery) -> nom de
# table Postgres reelle. "reel"/"post"/"event"/"concert" ont tous
# category/embedding/ai_analysis_status/status migres (cf.
# app/db/postgres/migrations.py cote stream_backend, 2026-08). "live" reste
# absent de ce mapping : pas de contenu fixe a analyser (stream temps reel
# via LiveKit), hors scope pour l'instant.
_TABLE_FOR_TYPE = {
    "reel": "reels",
    "post": "posts",
    "event": "events",
    "concert": "concerts",
}

# content_type ai_service -> ReportContentType cote stream_backend
# (app/db/postgres/models/report.py) — memes valeurs actuellement, mapping
# explicite plutot qu'une supposition d'egalite, pour rester correct si l'un
# des deux vocabulaires diverge un jour.
_REPORT_CONTENT_TYPE = {
    "reel": "reel",
    "post": "post",
    "event": "event",
    "concert": "concert",
}


def _table(content_type: str) -> str:
    return _TABLE_FOR_TYPE[content_type]


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
    table = _table(content_type)
    conn = await _connect()
    try:
        await conn.execute(
            f"UPDATE {table} SET embedding = $1 WHERE id = $2",
            embedding, content_id,
        )
    finally:
        await conn.close()


# normalize_category(None) (stream_backend/app/utils/content_category.py)
# retourne toujours "autre", jamais NULL, mais UNIQUEMENT pour les posts (seul
# type qui passe par cette fonction a la creation, cf. posts_router.py). Event
# et Concert n'ont pas d'equivalent : leur colonne `category` est
# nullable=True SANS defaut applicatif -> reste NULL tant que le createur n'a
# rien choisi (constate 2026-08-05 : un evenement/concert de test avait
# category=None apres une analyse IA reussie, alors que le meme post avait
# bien "tech" -- le WHERE category = 'autre' de save_category ne matchait
# jamais une ligne NULL). D'ou le double filtre ci-dessous : "autre" (posts/
# reels) OU NULL (events/concerts), jamais les deux en meme temps par type
# mais le code doit couvrir les deux sans avoir besoin de savoir lequel
# s'applique a quel content_type.
DEFAULT_CATEGORY = "autre"


async def save_category(content_type: str, content_id: str, category: str) -> None:
    """Ecrit la categorie inferee dans la colonne `category` existante.

    Ne remplace que `category = "autre"` OU `category IS NULL` — jamais une
    categorie deja choisie explicitement par le createur. Limite connue pour
    les posts/reels : impossible de distinguer en base un "autre" par defaut
    (createur n'a rien choisi) d'un "autre" choisi expres — les deux cas sont
    ecrases par l'IA de la meme facon. Compromis accepte faute d'une colonne
    separee (ex. category_is_explicit) qui n'existe pas dans le schema actuel."""
    table = _table(content_type)
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(
            f"UPDATE {table} SET category = $1 WHERE id = $2 AND (category = $3 OR category IS NULL)",
            category, content_id, DEFAULT_CATEGORY,
        )
    finally:
        await conn.close()


# Mapping palier de moderation -> statut de la ligne (reels/posts/events/
# concerts partagent tous les memes libelles "limited"/"archived" depuis
# 2026-08, cf. migrations.py cote stream_backend). "auto_remove" -> archived
# (pas de suppression definitive, cf. block_reported_content existant qui
# fait la meme chose pour les signalements humains -- reversible par un
# admin, contrairement a delete_reported_content).
_TIER_TO_STATUS = {"auto_remove": "archived", "limited": "limited"}

# Reason par defaut si `reasons` est vide malgre un tier != "none" -- ne
# devrait jamais arriver en pratique (build_verdict ne renvoie un tier positif
# que s'il y a au moins 1 reason), mais Report.reason est NOT NULL cote
# backend donc il faut une valeur de repli plutot que de planter l'ecriture.
_FALLBACK_REPORT_REASON = "other"


async def get_content_owner(content_type: str, content_id: str) -> str | None:
    """Retourne le user_id (str) du createur du contenu, ou None si introuvable.
    Utilise pour notifier le createur du resultat de sa propre analyse IA
    (cf. workers/tasks.py) -- les taches ai.analyze_* ne recoivent que
    l'id du contenu, pas l'auteur.

    La colonne portant l'auteur differe selon le type : user_id (reel/post),
    organizer_id (event), artist_id (concert)."""
    table = _table(content_type)
    owner_column = {
        "reel": "user_id", "post": "user_id",
        "event": "organizer_id", "concert": "artist_id",
    }[content_type]
    conn = await asyncpg.connect(settings.database_url)
    try:
        row = await conn.fetchrow(f"SELECT {owner_column} AS owner_id FROM {table} WHERE id = $1", content_id)
        return str(row["owner_id"]) if row else None
    finally:
        await conn.close()


async def mark_analysis_done(content_type: str, content_id: str) -> None:
    """Marque ai_analysis_status="done" (colonne UI uniquement) — fait dans
    tous les cas (tier="none" inclus), contrairement a apply_moderation_verdict
    qui ne fait rien si tier="none"."""
    table = _table(content_type)
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(
            f"UPDATE {table} SET ai_analysis_status = 'done' WHERE id = $1",
            content_id,
        )
    finally:
        await conn.close()


async def mark_cleared(content_type: str, content_id: str) -> None:
    """Transition pending_review -> published quand le verdict est "cleared"
    (tier="none", aucun signal detecte) — le vrai chainon manquant du flux
    synchrone-avant-publication (2026-08bis) : create_reel/create_post/
    create_event/create_concert (cote stream_backend) forcent desormais
    status="pending_review" (invisible de tous, y compris le createur, sauf
    dans sa file de verification GET /reports/me) des qu'un media est
    present, PRECISEMENT pour corriger un contenu explicite reste visible
    pendant la fenetre d'analyse asynchrone. Sans cette fonction, un contenu
    "cleared" resterait bloque invisible indefiniment.

    N'agit QUE si le statut actuel est encore pending_review (jamais
    n'ecrase limited/archived deja poses par un autre chemin, ni published
    deja atteint par ailleurs) -- WHERE conditionnel, pas de UPDATE aveugle.

    `published_at` n'existe que sur Event/Concert (colonne absente sur
    Reel/Post, verifie 2026-08bis) -- deux requetes distinctes plutot qu'un
    UPDATE generique qui planterait en SQL pour reel/post."""
    table = _table(content_type)
    has_published_at = content_type in ("event", "concert")
    conn = await asyncpg.connect(settings.database_url)
    try:
        if has_published_at:
            await conn.execute(
                f"UPDATE {table} SET status = 'published', published_at = now() "
                f"WHERE id = $1 AND status = 'pending_review'",
                content_id,
            )
        else:
            await conn.execute(
                f"UPDATE {table} SET status = 'published' WHERE id = $1 AND status = 'pending_review'",
                content_id,
            )
    finally:
        await conn.close()
    await _invalidate_content_cache(content_type, content_id)


async def apply_moderation_verdict(
    content_type: str,
    content_id: str,
    verdict: dict,
    reporter_id: str,
) -> None:
    """Applique le verdict de moderation_pipeline.build_verdict() en base :
    cree un Report (meme table que les signalements humains, reporter_id =
    compte systeme dedie) et, pour les paliers "limited"/"auto_remove",
    change le statut du contenu. Ne fait rien si tier == "none" (cf.
    mark_analysis_done pour le badge UI, mis a jour separement dans tous les cas).

    Jamais de suppression definitive (DELETE) : "auto_remove" passe le
    contenu a status="archived", reversible par un admin via l'interface de
    moderation existante -- coherent avec le principe deja pose de ne jamais
    laisser l'IA prendre une decision irreversible seule.

    Bug corrige (2026-08) : cette fonction ne changeait le statut QUE pour
    les reels avant l'extension a posts/events/concerts (`table == "reels"`
    code en dur), alors que le mapping table couvrait deja "post" -- jamais
    declenche en pratique faute d'appelant sur un autre type que reel.
    """
    tier = verdict.get("tier", "none")
    if tier == "none":
        return

    reasons = verdict.get("reasons") or [_FALLBACK_REPORT_REASON]
    table = _table(content_type)
    report_content_type = _REPORT_CONTENT_TYPE[content_type]
    new_status = _TIER_TO_STATUS.get(tier)

    conn = await asyncpg.connect(settings.database_url)
    try:
        async with conn.transaction():
            # Pas de contrainte unique en base sur (reporter_id, content_type,
            # content_id) — seule create_report() la fait cote applicatif pour
            # les signalements humains. Verification manuelle ici pour eviter
            # des doublons si analyze_* est relance (retry Celery, re-run
            # manuel) sur un contenu deja modere par ce meme compte systeme.
            already_reported = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM reports WHERE reporter_id = $1 AND content_type = $2 AND content_id = $3)",
                reporter_id, report_content_type, content_id,
            )
            if already_reported:
                return

            for reason in reasons:
                await conn.execute(
                    """
                    INSERT INTO reports (id, reporter_id, content_type, content_id, reason, details, status, created_at)
                    VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, now())
                    """,
                    reporter_id, report_content_type, content_id, reason,
                    f"Signale automatiquement par ai_service (palier: {tier})",
                    "resolved" if tier == "auto_remove" else "pending",
                )

            if new_status:
                # pending_review (2026-08bis) : le contenu est desormais
                # 'pending_review' (pas 'published') au moment ou ce verdict
                # arrive, puisque l'analyse tourne AVANT publication -- bug
                # constate 2026-08bis : cette clause ne matchait plus rien
                # (restait bloque en pending_review meme apres un verdict
                # limited/auto_remove), le WHERE devait couvrir les deux
                # statuts de depart possibles.
                await conn.execute(
                    f"UPDATE {table} SET status = $1 WHERE id = $2 AND status IN ('published', 'pending_review')",
                    new_status, content_id,
                )
    finally:
        await conn.close()

    if new_status:
        await _invalidate_content_cache(content_type, content_id)


async def find_similar(content_type: str, embedding: list[float], limit: int = 10) -> list[dict]:
    table = _table(content_type)
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
