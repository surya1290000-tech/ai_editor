"""
core/cache.py

Production Cache Validation & Stale Artifact Protection
────────────────────────────────────────────────────────
Guarantees that cached stage outputs are only reused when:
1. Source video SHA-256 matches exactly.
2. Pipeline version matches.
3. Model version matches.
4. Relevant config hash matches.

Never reuses artifacts solely because files exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

PIPELINE_VERSION = "3.5.0"


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hex digest of a file."""
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Cannot compute hash: file does not exist: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(config_data: Optional[Dict[str, Any]]) -> str:
    """Computes SHA-256 of canonical JSON serialized configuration."""
    if not config_data:
        return hashlib.sha256(b"{}").hexdigest()
    raw = json.dumps(config_data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class StageCacheManager:
    """
    Manages cryptographic identity and invalidation for pipeline stage artifacts.
    """

    def __init__(self, pipeline_version: str = PIPELINE_VERSION):
        self.pipeline_version = pipeline_version
        self._hash_cache: Dict[str, str] = {}

    def get_source_hash(self, source_path: Path | str) -> str:
        """Cached retrieval of source video SHA-256."""
        path_str = str(Path(source_path).resolve())
        if path_str not in self._hash_cache:
            self._hash_cache[path_str] = compute_file_sha256(path_str)
        return self._hash_cache[path_str]

    def is_stage_valid(
        self,
        stage_dir: Path | str,
        stage_name: str,
        expected_artifacts: list[str],
        source_video_path: Path | str,
        model_version: str = "default",
        config_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Validates that all expected artifacts exist AND the stage cache manifest
        matches source hash, pipeline version, model version, and config hash.
        """
        s_dir = Path(stage_dir).resolve()
        if not s_dir.exists():
            return False

        # 1. Verify all expected artifact files physically exist and are non-empty
        for art in expected_artifacts:
            art_p = s_dir / art
            if not art_p.exists() or art_p.stat().st_size == 0:
                return False

        # 2. Check manifest file
        manifest_p = s_dir / f".cache_{stage_name}.json"
        if not manifest_p.exists():
            return False

        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            source_hash = self.get_source_hash(source_video_path)
            config_hash = compute_config_hash(config_data)

            if manifest.get("source_video_sha256") != source_hash:
                return False
            if manifest.get("pipeline_version") != self.pipeline_version:
                return False
            if manifest.get("model_version") != model_version:
                return False
            if manifest.get("config_hash") != config_hash:
                return False

            return True
        except Exception:
            return False

    def write_stage_manifest(
        self,
        stage_dir: Path | str,
        stage_name: str,
        source_video_path: Path | str,
        model_version: str = "default",
        config_data: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Writes .cache_<stage_name>.json capturing full provenance.
        """
        s_dir = Path(stage_dir).resolve()
        s_dir.mkdir(parents=True, exist_ok=True)
        manifest_p = s_dir / f".cache_{stage_name}.json"

        source_hash = self.get_source_hash(source_video_path)
        config_hash = compute_config_hash(config_data)

        payload = {
            "stage_name": stage_name,
            "source_video_sha256": source_hash,
            "source_video_name": Path(source_video_path).name,
            "pipeline_version": self.pipeline_version,
            "model_version": model_version,
            "config_hash": config_hash,
            "extra_metadata": extra_meta or {},
        }

        with open(manifest_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        return manifest_p
