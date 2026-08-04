from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    rabbitmq_url: str
    redis_url: str

    text_model_path: str
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    yolo_model_path: str
    whisper_model_size: str = "base"

    max_ram_mb_for_models: int = 2500
    ai_queue_concurrency: int = 1

    # UUID du compte systeme stream_backend (users.id) utilise comme reporter_id
    # pour les Report crees automatiquement par la moderation IA (cf.
    # moderation_pipeline.py, workers/tasks.py::_apply_moderation_verdict).
    # Cree une seule fois en base (email ai-moderation@gofolyx.internal),
    # voir ARCHITECTURE_IA.md section 9bis pour le detail.
    ai_moderation_reporter_id: str


settings = Settings()
