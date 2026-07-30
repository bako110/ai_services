"""Benchmark du LLM texte (Qwen3 GGUF via llama.cpp), via le vrai
text_pipeline.classify_content() — pas un prompt duplique ici, pour ne
jamais diverger du code reellement utilise en prod.

Usage:
    python scripts/bench_text.py --model models/qwen3-1.7b-instruct-q4_k_m.gguf

Go/no-go (ARCHITECTURE_IA.md section 7) : si ram_delta_mb depasse largement
MAX_RAM_MB_FOR_MODELS (.env) ou si l'inference depasse quelques secondes,
retomber sur un modele plus petit (1.7B au lieu de 4B) ou des regles
mots-cles simples pour le tagging texte.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_common import measure, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="chemin vers le .gguf")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    os.environ["TEXT_MODEL_PATH"] = args.model

    from app.pipelines import text_pipeline

    results = {}
    with measure("chargement + inference (1er appel)") as m_first:
        result = text_pipeline.classify_content(
            title="Je cuisine un plat traditionnel avec ma grand-mere",
            description="recette de famille",
        )
    results["chargement_et_inference"] = m_first

    print("--- reponse (parsee) ---")
    print(result)
    print("--- mesures ---")
    report(results)


if __name__ == "__main__":
    main()
