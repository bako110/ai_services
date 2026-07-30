from celery import Celery

from app.core.config import settings

# Reutilise le meme RabbitMQ/Redis que stream_backend (app/tasks/celery_app.py
# cote backend) — pas de second broker a faire tourner sur un VPS a RAM limitee.
# Isolation : queue "ai" dediee, jamais les queues de stream_backend.
celery_app = Celery(
    "ai_service",
    broker=settings.rabbitmq_url,
    backend=settings.redis_url,
)

# concurrency=1 au lancement du worker
# (celery -A app.workers.celery_app worker -Q ai --concurrency=1)
# — voir ARCHITECTURE_IA.md section 4 pour la justification.
celery_app.conf.task_default_queue = "ai"
