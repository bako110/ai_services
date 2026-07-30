"""Benchmark OpenCLIP (embedding d'image).

Usage:
    python scripts/bench_image.py --image sample.jpg
"""

import argparse

from bench_common import measure, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    import open_clip
    import torch
    from PIL import Image

    results = {}
    with measure("chargement modele") as m_load:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, pretrained=args.pretrained
        )
        model.eval()
    results["load"] = m_load

    with measure("inference (1 image)") as m_infer:
        image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(image)
    results["inference"] = m_infer

    print(f"--- embedding dimension: {features.shape[-1]} ---")
    print("--- mesures ---")
    report(results)


if __name__ == "__main__":
    main()
