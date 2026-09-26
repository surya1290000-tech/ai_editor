"""
core/json_utils.py

JSON serialization utilities that cleanly handle NumPy 2.x types (bool, int, float, ndarray)
and Path objects without throwing TypeError.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def json_default(obj):
    """Fallback encoder for types not natively handled by json."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    # Check if object has numpy-like name or methods
    cls_name = obj.__class__.__name__
    if "bool" in cls_name.lower():
        return bool(obj)
    if "float" in cls_name.lower():
        return float(obj)
    if "int" in cls_name.lower():
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def dump_json(obj, fp, indent=2, ensure_ascii=False, **kwargs) -> None:
    """Serialize obj to JSON and write to open file pointer fp."""
    json.dump(obj, fp, indent=indent, ensure_ascii=ensure_ascii, default=json_default, **kwargs)


def dumps_json(obj, indent=2, ensure_ascii=False, **kwargs) -> str:
    """Serialize obj to JSON string."""
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=json_default, **kwargs)


def save_json(obj, path: Path | str, indent=2, ensure_ascii=False) -> None:
    """Write obj directly to file path safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii, default=json_default)


def load_json(path: Path | str) -> dict:
    """Load JSON from file path safely."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

