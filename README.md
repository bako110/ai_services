# ai_service

Squelette du microservice IA GoFolyX. Voir [`../ARCHITECTURE_IA.md`](../ARCHITECTURE_IA.md)
pour la conception complete avant de modifier quoi que ce soit ici.

Etat (2026-07-30) : **teste reellement sur le VPS de prod** (bench CLIP/YOLO/Qwen3
avec les vrais modeles, voir "Historique des problemes reels" plus bas) — pas
encore branche a stream_backend (aucun appel `send_task` cote backend).

Verifie contre le vrai code de `stream_backend` (2026-07-29) : tables
`reels`/`posts`/`lives` et colonne `category` existent deja (voir
`stream_backend/app/db/postgres/models/reel.py` etc.), la taxonomie
correspond a `stream_backend/app/utils/content_category.py`.

## Topologie de deploiement (2026-07-30) — serveur separe du backend

`ai_service` tourne sur son **propre VPS** (169.58.100.82), distinct de celui
du backend (178.238.230.82, deja dense : Postgres/Redis/RabbitMQ/Elasticsearch/
stream_app, ~3,5 Gi RAM libres). Les deux VPS sont relies par un **tunnel
WireGuard prive** :
- VPS backend = peer `10.10.0.1`, expose `postgres:5432`/`redis:6379`/`rabbitmq:5672`
  UNIQUEMENT sur cette IP privee (jamais publiquement — verifie par scan externe).
- VPS ai_service = peer `10.10.0.2`.

`docker-compose.yml` (racine) correspond a ce mode distant : pas de reseau
Docker partage, `.env.ai` contient des URLs completes pointant sur `10.10.0.1`.
`docker-compose.colocated.yml` + `.env.ai.colocated.example` restent pour
reference si l'architecture redevient un jour colocalisee (meme serveur que
le backend, reseau Docker partage `stream_net`).

## Installation — deux modes

**Dev local sans Docker** (test rapide sur un poste de dev, pas le VPS) :
```bash
python -m venv venv
venv\Scripts\activate          # Windows (dev)
# source venv/bin/activate     # Linux
pip install -r requirements.txt
copy .env.example .env         # renseigner DATABASE_URL/RABBITMQ_URL/REDIS_URL + chemins modeles
```

**Deploiement VPS distant (Docker, mode reel)** — sur 169.58.100.82, connecte
au backend via WireGuard :
```bash
cp .env.ai.example .env.ai     # completer avec les vrais mots de passe (10.10.0.1)
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

## Historique des problemes reels constates (2026-07-30, premier run sur VPS)

Deux incompatibilites de version decouvertes en testant avec les VRAIS
modeles (pas seulement en relisant le code) — corrigees dans `requirements.txt` :
- `ultralytics==8.2.18` ne connait pas l'architecture YOLO11 (bloc `C3k2`) —
  `torch.load` d'un poids `yolo11n.pt` reel echouait avec `AttributeError:
  Can't get attribute 'C3k2'`. Corrige vers `8.3.40`.
- `llama-cpp-python==0.2.76` ne connait pas l'architecture GGUF `qwen3` —
  chargement d'un vrai `Qwen3-1.7B-Q4_K_M.gguf` echouait avec `unknown model
  architecture: 'qwen3'`. Corrige vers `0.3.34` (wheel CPU precompilee,
  toujours pas de compilation source).

Chiffres reels mesures sur le VPS backend colocalise 178.238.230.82
(bench_moderation.py / bench_text.py, image de test neutre) :
- Chargement CLIP (ViT-B-32) : ~24s, +1,44 Go RAM (one-shot par job).
- `classify_nsfw` une fois CLIP charge : +0,24s, +5 Mo — confirme le cout
  quasi nul de la moderation NSFW une fois l'embedding recommandation deja
  calcule.
- YOLO (decharge CLIP, charge YOLO) : +2,9s, +57 Mo.
- Qwen3-1.7B-Q4_K_M : chargement ~3,8s +2 Go RAM, inference ~1,9s (~35 tok/s).

**Ecart de perf CPU entre les deux VPS Contabo (2026-07-30).** Sur le VPS
distant 169.58.100.82 (meme CPU annonce — AMD EPYC, 6 vCPU — et memes
modeles), l'inference Qwen3 mesure ~10s au lieu de ~1,9s (5x plus lent),
chargement CLIP ~33s au lieu de ~24s. RAM et threads identiques, mesure
reproductible sur 2 runs consecutifs (pas un effet de demarrage a froid).
Cause probable : vCPU plus contendus (sur-souscription differente entre les
deux offres/regions Contabo), pas un probleme de configuration cote
`ai_service`. A garder en tete pour le dimensionnement final — si cette
latence est genante en usage reel, verifier le type d'instance exact aupres
de Contabo avant d'incriminer le code.

## Campagne de tests qualite — faux positifs/negatifs (2026-07-31)

Suite a un premier constat alarmant (le prompt Qwen3 combine categorie+
moderation echouait sur 8/8 cas de test, cf. section suivante), une campagne
de tests structuree a ete menee sur les 3 signaux de moderation avant de
faire confiance au systeme. Echantillons : contenu neutre varie (cuisine,
sport, musique, mode, plage) + cas explicitement problematiques (arnaque,
haine, violence, spam), tous legaux/publics (aucune image NSFW reelle
utilisee — cf. decision produit de ne pas manipuler ce type de contenu pour
les tests).

**Moderation texte (Qwen3, apres fix du prompt double-appel — voir section
suivante)** : 5/5 cas problematiques detectes (arnaque evidente, arnaque
subtile, haine, violence, spam), 0/3 faux positif sur les cas neutres.
Categorie (musique/sport/cuisine...) reste peu fiable independamment de la
moderation — pas bloquant pour la moderation elle-meme.

**NSFW image (CLIP zero-shot)** : 5/6 images correctement `safe` (sport,
mode, cuisine, plage, photo generique) avec confidence sous le seuil de flag
(0.8) dans tous les cas — 0 flag errone declenche sur cet echantillon. Une
image (couteau de cuisine) reste mal etiquetee `sexual_explicit` en label
brut mais sous le seuil, donc sans consequence pratique pour l'instant (cf.
limite deja documentee du classifieur zero-shot).

**Objets sensibles (YOLO)** : 0 faux positif constate, y compris sur une
photo de cuisine avec pomme/bol/tasse (aucun couteau visible => pas de flag)
— la crainte initiale que "toute video de cuisine avec un couteau" declenche
un flag violence n'est pas confirmee empiriquement : YOLO ne detecte que la
presence reelle d'un couteau dans le cadre, pas la categorie "cuisine" en
general.

**Limite de cette campagne : echantillon petit (6 images, 8 textes), pas une
validation statistique rigoureuse.** Un vrai taux de faux positifs/negatifs
fiable demanderait des centaines d'exemples reels de la plateforme. Ces tests
donnent une premiere confiance raisonnable, pas une garantie.

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
