"""Load OmegaConf YAML configs with `defaults:` merge."""

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

    cfg = OmegaConf.load(path)
    defaults = cfg.get("defaults")
    if defaults:
        merged = OmegaConf.create({})
        for item in defaults:
            if isinstance(item, str):
                base = load_config(item, config_dir)
                merged = OmegaConf.merge(merged, base)
            elif isinstance(item, dict) and "_self_" in item:
                continue
        merged = OmegaConf.merge(merged, cfg)
        if "defaults" in merged:
            del merged["defaults"]
        return merged
    return cfg
