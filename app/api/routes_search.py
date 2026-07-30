from fastapi import APIRouter

from app.pipelines import image_pipeline
from app.vector import pgvector_store

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/similar/{content_type}/{content_id}")
async def similar_content(content_type: str, content_id: str, limit: int = 10):
    """Recherche par similarite d'embedding — necessite Phase 3 (cf. ARCHITECTURE_IA.md section 7)."""
    raise NotImplementedError("Phase 3 — pas encore implemente")
