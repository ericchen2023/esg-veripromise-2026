"""Lightweight YAML config loader with `extends` inheritance."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in over.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    parent = cfg.get("extends")
    if parent:
        parent_path = (p.parent / parent).resolve()
        base = load_config(parent_path)
        cfg = _deep_merge(base, cfg)
    return cfg
