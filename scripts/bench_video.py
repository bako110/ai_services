"""Benchmark YOLOv11n (detection d'objets sur une frame) + extraction FFmpeg.

Usage:
    python scripts/bench_video.py --video sample.mp4 --yolo-model models/yolov11n.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_common import measure, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--yolo-model", required=True)
    args = parser.parse_args()

    from app.pipelines.video_pipeline import extract_keyframes

    results = {}
    with measure("extraction frames (ffmpeg)") as m_extract:
        frames = extract_keyframes(args.video, count=3)
    results["extract"] = m_extract

    if not frames:
        print("Aucune frame extraite — verifier que ffmpeg est installe et le chemin video valide.")
        return

    from ultralytics import YOLO

    with measure("chargement modele YOLO") as m_load:
        model = YOLO(args.yolo_model)
    results["load"] = m_load

    with measure("inference (1 frame)") as m_infer:
        yolo_results = model.predict(frames[0], verbose=False)
    results["inference"] = m_infer

    labels = sorted({
        r.names[int(box.cls[0])]
        for r in yolo_results
        for box in r.boxes
        if float(box.conf[0]) >= 0.4
    })
    print(f"--- objets detectes: {labels} ---")
    print("--- mesures ---")
    report(results)


if __name__ == "__main__":
    main()
