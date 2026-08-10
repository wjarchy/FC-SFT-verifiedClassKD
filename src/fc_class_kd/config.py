"""Load OmegaConf YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(config_name: str = "base", config_dir: Path | None = None) -> Any:
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
    path = config_dir / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)
