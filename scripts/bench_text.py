"""Benchmark du LLM texte (Qwen3 GGUF via llama.cpp).

Usage:
    python scripts/bench_text.py --model models/qwen3-1.7b-instruct-q4_k_m.gguf

Go/no-go (ARCHITECTURE_IA.md section 7) : si ram_delta_mb depasse largement
MAX_RAM_MB_FOR_MODELS (.env) ou si l'inference depasse quelques secondes,
retomber sur un modele plus petit (1.7B au lieu de 4B) ou des regles
mots-cles simples pour le tagging texte.
"""

import argparse

from bench_common import measure, report

SAMPLE_PROMPT = (
    "Tu classes du contenu pour une app de streaming. "
    "Categories autorisees (choisis-en une seule, exactement): musique, sport, gaming, "
    "humour, danse, cuisine, mode, beaute, tech, education, lifestyle, art, voyage, "
    "business, actualite, spiritualite, famille, sante, autre.\n"
    "Contenu:\nJe cuisine un plat traditionnel avec ma grand-mere, recette de famille.\n\n"
    "Reponds uniquement au format: categorie|confiance(0-1)|signale(oui/non)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="chemin vers le .gguf")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    from llama_cpp import Llama

    results = {}
    with measure("chargement modele") as m_load:
        llm = Llama(model_path=args.model, n_ctx=2048, n_threads=args.threads)
    results["load"] = m_load

    with measure("inference (1 prompt)") as m_infer:
        output = llm(SAMPLE_PROMPT, max_tokens=32, temperature=0.0)
    results["inference"] = m_infer

    print("--- reponse modele ---")
    print(output["choices"][0]["text"].strip())
    print("--- mesures ---")
    report(results)


if __name__ == "__main__":
    main()
