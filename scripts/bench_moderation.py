"""Benchmark de la moderation d'image (NSFW zero-shot CLIP + objets sensibles YOLO).

Reutilise les memes modeles que bench_image.py/bench_video.py — l'interet ici
est de mesurer le cout ADDITIONNEL de la classification NSFW par rapport a un
simple embed_image() deja fait pour la recommandation (cout attendu ~nul,
puisque c'est le meme forward CLIP + un encode_text sur une poignee de
prompts, cf. app/pipelines/image_pipeline.py::classify_nsfw).

Usage:
    python scripts/bench_moderation.py --image sample.jpg --yolo-model models/yolov11n.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_common import measure, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--yolo-model",
        help="Doit correspondre a YOLO_MODEL_PATH dans .env : "
        "detect_sensitive_objects() charge via app.core.config.settings, "
        "pas cet argument (contrairement a bench_video.py qui charge YOLO directement).",
    )
    args = parser.parse_args()

    if args.yolo_model:
        import os

        os.environ["YOLO_MODEL_PATH"] = args.yolo_model

    from app.pipelines.image_pipeline import classify_nsfw, embed_image
    from app.pipelines.video_pipeline import detect_sensitive_objects

    results = {}

    with measure("embed_image (baseline recommandation)") as m_embed:
        embed_image(args.image)
    results["embed_baseline"] = m_embed

    with measure("classify_nsfw (cout additionnel attendu ~0)") as m_nsfw:
        nsfw = classify_nsfw(args.image)
    results["classify_nsfw"] = m_nsfw

    with measure("detect_sensitive_objects (YOLO, modele different -> dechargement CLIP)") as m_yolo:
        sensitive_objects = detect_sensitive_objects(args.image)
    results["detect_sensitive_objects"] = m_yolo

    print(f"--- nsfw: {nsfw} ---")
    print(f"--- objets sensibles: {sensitive_objects} ---")
    print("--- mesures ---")
    report(results)


if __name__ == "__main__":
    main()
