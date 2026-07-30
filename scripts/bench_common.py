"""Utilitaires partages par les scripts bench_*.py (Phase 0, ARCHITECTURE_IA.md section 7).

Objectif : mesurer, sur la machine reelle (VPS Contabo, pas un poste de dev),
la RAM consommee et la latence de chaque modele, pour decider en connaissance
de cause quelle taille de modele est soutenable — pas de suppositions.

Usage typique dans un script bench_x.py :

    from bench_common import measure

    with measure("chargement modele") as m:
        model = load_model()
    with measure("inference") as m2:
        result = model.run(sample)
    report({"load": m, "inference": m2})
"""

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import psutil

_process = psutil.Process()


def _rss_mb() -> float:
    return _process.memory_info().rss / (1024 * 1024)


@dataclass
class Measurement:
    label: str
    duration_s: float
    ram_before_mb: float
    ram_after_mb: float
    ram_delta_mb: float


@contextmanager
def measure(label: str):
    ram_before = _rss_mb()
    start = time.perf_counter()
    holder: dict = {}
    yield holder
    duration = time.perf_counter() - start
    ram_after = _rss_mb()
    holder["result"] = Measurement(
        label=label,
        duration_s=round(duration, 2),
        ram_before_mb=round(ram_before, 1),
        ram_after_mb=round(ram_after, 1),
        ram_delta_mb=round(ram_after - ram_before, 1),
    )


def report(measurements: dict) -> None:
    payload = {name: asdict(holder["result"]) for name, holder in measurements.items()}
    payload["system_total_ram_mb"] = round(psutil.virtual_memory().total / (1024 * 1024), 1)
    payload["system_available_ram_mb"] = round(psutil.virtual_memory().available / (1024 * 1024), 1)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
