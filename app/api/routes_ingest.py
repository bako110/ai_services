from fastapi import APIRouter

from app.workers.tasks import analyze_reel

router = APIRouter(prefix="/analyze", tags=["ingest"])


@router.post("/reel/{reel_id}")
def trigger_reel_analysis(reel_id: str, video_path: str, title: str = "", description: str = ""):
    """Enfile une tache Celery, ne bloque jamais l'appelant (cf. ARCHITECTURE_IA.md section 5)."""
    task = analyze_reel.delay(reel_id, video_path, title, description)
    return {"task_id": task.id, "status": "queued"}
