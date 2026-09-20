from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = config_path.resolve().parent

    def resolve(value: str) -> str:
        candidate = Path(value).expanduser()
        return str(candidate if candidate.is_absolute() else (base / candidate).resolve())

    config["lmdb_path"] = resolve(config["lmdb_path"])
    config["index_paths"] = {
        name: resolve(value) for name, value in config["index_paths"].items()
    }
    config["mask_paths"] = {
        name: {setting: resolve(value) for setting, value in masks.items()}
        for name, masks in config.get("mask_paths", {}).items()
    }
    config["_path"] = str(config_path.resolve())
    return config
