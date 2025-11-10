from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class ArtifactBundle:
    """Simple container that stores serialized model-related files in one folder."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, obj: Any, name: str) -> Path:
        path = self.base_dir / f"{name}.joblib"
        joblib.dump(obj, path)
        return path

    def load(self, name: str) -> Any:
        path = self.base_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        return joblib.load(path)

    def save_metadata(self, data: dict, filename: str = "meta.json") -> Path:
        path = self.base_dir / filename
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        return path

    def load_metadata(self, filename: str = "meta.json") -> dict:
        path = self.base_dir / filename
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
