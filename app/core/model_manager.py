"""Charge un seul modele lourd a la fois en memoire.

Contrainte serveur (voir ARCHITECTURE_IA.md section 2) : ~3,5 Gi RAM libres,
pas de GPU. Charger OpenCLIP + YOLO + Whisper + un LLM simultanement n'est pas
soutenable. Ce manager garantit qu'un seul modele "lourd" est resident a la
fois : avant de charger le suivant, il decharge le precedent.

Pas d'appel direct aux modeles ailleurs dans le code — toujours passer par
`ModelManager.get(name, loader)`.
"""

import gc
from typing import Callable, Optional


class ModelManager:
    def __init__(self) -> None:
        self._current_name: Optional[str] = None
        self._current_model = None

    def get(self, name: str, loader: Callable[[], object]) -> object:
        if self._current_name == name and self._current_model is not None:
            return self._current_model

        self._unload_current()
        self._current_model = loader()
        self._current_name = name
        return self._current_model

    def _unload_current(self) -> None:
        if self._current_model is None:
            return
        del self._current_model
        self._current_model = None
        self._current_name = None
        gc.collect()


model_manager = ModelManager()
