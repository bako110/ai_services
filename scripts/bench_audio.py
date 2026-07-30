"""Benchmark faster-whisper (transcription audio).

Usage:
    python scripts/bench_audio.py --audio sample.wav --size base

Go/no-go : si la transcription est plus lente que la duree reelle de l'audio
(facteur temps-reel > 1x), un traitement batch differe reste viable, mais un
usage "quasi temps reel" (heartbeat live) ne l'est pas avec ce size de modele.
"""

import argparse
import wave

from bench_common import measure, report


def _audio_duration_s(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as f:
            return f.getnframes() / f.getframerate()
    except (wave.Error, EOFError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--size", default="base", choices=["tiny", "base", "small", "medium"])
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    results = {}
    with measure("chargement modele") as m_load:
        model = WhisperModel(args.size, device="cpu", compute_type="int8")
    results["load"] = m_load

    with measure("transcription") as m_infer:
        segments, _info = model.transcribe(args.audio, beam_size=1)
        text = " ".join(s.text.strip() for s in segments)
    results["inference"] = m_infer

    duration = _audio_duration_s(args.audio)
    if duration:
        realtime_factor = m_infer["result"].duration_s / duration
        print(f"--- duree audio: {duration:.1f}s, facteur temps-reel: {realtime_factor:.2f}x ---")

    print(f"--- transcription ---\n{text}")
    print("--- mesures ---")
    report(results)


if __name__ == "__main__":
    main()
