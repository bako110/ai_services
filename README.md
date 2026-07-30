# ai_service

Squelette du microservice IA GoFolyX. Voir [`../ARCHITECTURE_IA.md`](../ARCHITECTURE_IA.md)
pour la conception complete avant de modifier quoi que ce soit ici.

Etat actuel : **structure de code, pas encore teste sur donnees reelles.**
Ne pas deployer en prod avant la Phase 0 (preuve de concept, mesure RAM/latence
sur le VPS reel) decrite dans ARCHITECTURE_IA.md section 7.

Verifie contre le vrai code de `stream_backend` (2026-07-29) : tables
`reels`/`posts`/`lives` et colonne `category` existent deja (voir
`stream_backend/app/db/postgres/models/reel.py` etc.), la taxonomie
correspond a `stream_backend/app/utils/content_category.py`. Broker Celery
et format DATABASE_URL corriges pour matcher `stream_backend/app/tasks/celery_app.py`
et `stream_backend/app/config.py` — **ai_service reutilise le meme RabbitMQ/Redis**,
pas de second broker.

## Installation — deux modes

**Dev local sans Docker** (test rapide sur un poste de dev, pas le VPS) :
```bash
python -m venv venv
venv\Scripts\activate          # Windows (dev)
# source venv/bin/activate     # Linux
pip install -r requirements.txt
copy .env.example .env         # renseigner DATABASE_URL/RABBITMQ_URL/REDIS_URL + chemins modeles
```

**Deploiement VPS (Docker, mode reel)** — rejoint le reseau Docker existant de
`stream_backend` pour reutiliser Postgres/RabbitMQ/Redis deja en place, sans
dupliquer les secrets :
```bash
cp .env.ai.example .env.ai     # variables sans secret (chemins modeles)
docker network ls              # verifier le nom reel du reseau stream_net
docker compose up -d --build   # voir docker-compose.yml
```

Modeles a telecharger manuellement (pas commit dans le repo — trop volumineux) :
- `models/qwen3-1.7b-instruct-q4_k_m.gguf` (Hugging Face, format GGUF)
- `models/yolov11n.pt` (Ultralytics)
- OpenCLIP et faster-whisper telechargent leurs poids automatiquement au premier
  appel (cache local `~/.cache`)

## Phase 0 — benchmarks (a faire en premier, sur le VPS reel)

Avant d'ecrire la moindre integration avec `stream_backend`, valider que
chaque modele tient dans le budget RAM/latence du serveur reel (11 Gi total,
~3,5 Gi libres observes, CPU only) :

```bash
python scripts/bench_text.py --model models/qwen3-1.7b-instruct-q4_k_m.gguf
python scripts/bench_image.py --image sample.jpg
python scripts/bench_audio.py --audio sample.wav --size base
python scripts/bench_video.py --video sample.mp4 --yolo-model models/yolov11n.pt
python scripts/bench_moderation.py --image sample.jpg --yolo-model models/yolov11n.pt
```

Chaque script imprime un JSON avec `ram_delta_mb` (RAM ajoutee par le
chargement/l'inference) et `duration_s`. Criteres go/no-go :
- **RAM** : la somme des `ram_delta_mb` de chargement de tous les modeles ne
  doit jamais approcher les ~3,5 Gi disponibles — en pratique, comme un seul
  modele est charge a la fois (`model_manager`), c'est le `ram_delta_mb` du
  plus gros modele individuel qui compte, pas la somme.
- **Texte (Qwen3)** : si `ram_delta_mb` du chargement depasse ~1,5-2 Go ou si
  l'inference depasse quelques secondes, retomber sur la variante 1.7B (au
  lieu de 4B) voire sur des regles mots-cles simples pour un premier jet.
- **Audio (faster-whisper)** : si le facteur temps-reel affiche par
  `bench_audio.py` est > 1x, le traitement doit rester strictement batch/differe
  (jamais un heartbeat live "quasi temps reel").
- Si un modele echoue ces criteres, documenter la decision (modele retenu +
  raison) dans `ARCHITECTURE_IA.md` section 3 avant de continuer.

## Lancer en dev

```bash
uvicorn app.main:app --reload --port 8100
celery -A app.workers.celery_app worker -Q ai --concurrency=1 --loglevel=info
```

## Moderation de contenu (2026-07-30)

`app/pipelines/moderation_pipeline.py` combine trois signaux en un verdict
unique (`build_verdict`), sans jamais agir lui-meme (pas de suppression, pas
d'ecriture DB) :
- **NSFW** (`image_pipeline.classify_nsfw`) : classification zero-shot par
  similarite CLIP (embedding image vs. prompts texte "explicite"/"suggestif"/
  "safe") — reutilise le meme modele CLIP que `embed_image()`, donc **cout RAM
  quasi nul** en plus de ce qui tourne deja pour la recommandation. Pas de
  modele NSFW dedie supplementaire a charger (voir ARCHITECTURE_IA.md §9 pour
  la justification du choix).
- **Objets sensibles** (`video_pipeline.detect_sensitive_objects`) : filtre
  les detections YOLOv11n deja calculees sur `{"knife", "scissors"}` — les
  seules classes COCO pertinentes. **YOLOv11n seul ne detecte ni armes a feu
  ni sang** (absents du jeu d'entrainement COCO), documente explicitement
  dans le code pour ne pas laisser croire a une couverture qu'il n'a pas.
- **Texte** (`text_pipeline.classify_content`) : le LLM Qwen3 produit
  desormais un `flag_reason` explicite (spam/arnaque/haine/violence/aucun)
  aligne sur `ReportReason` de `stream_backend`, au lieu d'un simple booleen
  sans contexte.

`workers/tasks.py::analyze_reel` calcule ce verdict et le retourne dans le
resultat de la tache Celery, mais **n'ecrit rien en base** — aucune colonne
ni table dediee n'existe encore cote `stream_backend` pour stocker un verdict
d'origine IA (le systeme `reports` existant suppose un `reporter_id` humain).
Brancher ça necessite une decision cote `stream_backend`, volontairement hors
scope ici (consigne explicite : ne pas modifier `stream_backend` sans
validation prealable).

## Avant de coder une nouvelle fonctionnalite

1. Le modele doit passer par `app.core.model_manager.model_manager` — jamais
   de chargement direct dans un pipeline, sinon le budget RAM du serveur explose.
2. Toute tache lourde est une tache Celery sur la queue `ai`, jamais un appel
   synchrone dans une route FastAPI.
3. Un echec de pipeline (frame manquante, audio silencieux, modele indisponible)
   ne doit jamais lever d'exception non geree — retourner un resultat partiel
   ou `None`, cf. principe de degradation gracieuse d'ARCHITECTURE.md.
