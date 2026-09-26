"""
agents/agent3_creative/assets.py

Asset Resolver Interface for Agent 3 (Creative Director)
────────────────────────────────────────────────────────
Resolves conceptual B-roll queries and SFX cues against the local asset library
or external provider adapters.

Strict rule:
If no asset exists on disk:
  status = UNRESOLVED_ASSET
Never hallucinate or fake completion.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.logger import get_logger

logger = get_logger(__name__)


class BaseAssetResolver(ABC):
    """Abstract interface for B-roll and SFX asset resolution."""

    @abstractmethod
    def resolve_broll(
        self,
        query: str,
        required_duration: float,
        role: str = "context",
    ) -> dict[str, Any]:
        """
        Attempt to resolve a B-roll request against available assets.
        Returns dict with status ("RESOLVED" or "UNRESOLVED_ASSET"), asset_path, etc.
        """
        pass

    @abstractmethod
    def resolve_sfx(
        self,
        category: str,
        sound_cue: str,
    ) -> dict[str, Any]:
        """
        Attempt to resolve an SFX request against available library assets.
        Returns dict with status ("RESOLVED" or "UNRESOLVED_ASSET"), asset_path, etc.
        """
        pass


class LocalAssetLibraryResolver(BaseAssetResolver):
    """
    Resolves B-roll and SFX from local workspace asset directories:
    - assets/user_broll/
    - assets/broll/
    - assets/sfx/
    """

    SUPPORTED_SFX_CATEGORIES = {
        "whoosh": ["whoosh_fast.wav", "whoosh_soft.wav", "whoosh.mp3", "sfx_whoosh.wav"],
        "pop": ["pop_bubble.wav", "pop_click.wav", "pop.mp3", "sfx_pop.wav"],
        "impact": ["impact_cinematic.wav", "impact_subtle.wav", "impact.mp3", "sfx_impact.wav"],
        "riser": ["riser_tension.wav", "riser_short.wav", "riser.mp3", "sfx_riser.wav"],
        "subtle_hit": ["subtle_hit.wav", "accent_hit.wav", "sfx_subtle_hit.wav"],
    }

    def __init__(self, workspace_root: Optional[Path | str] = None):
        if workspace_root:
            self.root = Path(workspace_root).resolve()
        else:
            self.root = Path(__file__).resolve().parent.parent.parent

        self.broll_dirs = [
            self.root / "assets" / "user_broll",
            self.root / "assets" / "broll",
        ]
        self.sfx_dirs = [
            self.root / "assets" / "sfx",
            self.root / "assets" / "audio",
        ]

    def resolve_broll(
        self,
        query: str,
        required_duration: float,
        role: str = "context",
    ) -> dict[str, Any]:
        """
        Searches local B-roll directories for video files matching keywords in query.
        If no matching physical asset is found, strictly reports UNRESOLVED_ASSET.
        """
        query_terms = [t.lower() for t in query.split() if len(t) > 2]

        if not hasattr(self, "_used_broll"):
            self._used_broll = set()

        matches = []
        for b_dir in self.broll_dirs:
            if not b_dir.exists() or not b_dir.is_dir():
                continue

            for f in b_dir.iterdir():
                if f.is_file() and f.suffix.lower() in [".mp4", ".mov", ".mkv", ".webm"]:
                    name_lower = f.stem.lower()
                    if any(term in name_lower for term in query_terms):
                        matches.append(f)

        if matches:
            # Prefer an unused match to avoid repeating the identical clip
            chosen = next((m for m in matches if str(m) not in self._used_broll), matches[0])
            self._used_broll.add(str(chosen))
            logger.info(f"AssetResolver | Resolved B-roll '{chosen.name}' for query '{query}'")
            return {
                "status": "RESOLVED",
                "asset_path": str(chosen.resolve()),
                "asset_name": chosen.name,
                "query": query,
                "role": role,
                "required_duration": required_duration,
            }

        logger.info(f"AssetResolver | No local asset found for B-roll query: '{query}' -> UNRESOLVED_ASSET")
        return {
            "status": "UNRESOLVED_ASSET",
            "asset_path": None,
            "asset_name": None,
            "query": query,
            "role": role,
            "required_duration": required_duration,
        }

    def resolve_sfx(
        self,
        category: str,
        sound_cue: str,
    ) -> dict[str, Any]:
        """
        Searches local SFX directories for audio files matching category or cue.
        If no physical asset is found, strictly reports UNRESOLVED_ASSET.
        """
        clean_cat = category.lower().replace("sfx_", "").strip()
        cue_lower = sound_cue.lower()

        # Check candidate filenames
        candidates = self.SUPPORTED_SFX_CATEGORIES.get(clean_cat, [])

        for s_dir in self.sfx_dirs:
            if not s_dir.exists() or not s_dir.is_dir():
                continue

            # 1. Check for specific cue match
            for f in s_dir.iterdir():
                if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".aac", ".flac", ".ogg"]:
                    f_stem = f.stem.lower()
                    if cue_lower in f_stem or clean_cat in f_stem:
                        logger.info(f"AssetResolver | Resolved SFX '{f.name}' for cue '{sound_cue}'")
                        return {
                            "status": "RESOLVED",
                            "category": clean_cat,
                            "sound_cue": sound_cue,
                            "asset_path": str(f.resolve()),
                            "asset_name": f.name,
                        }

            # 2. Check standard candidate names
            for candidate in candidates:
                cand_path = s_dir / candidate
                if cand_path.exists():
                    logger.info(f"AssetResolver | Resolved standard SFX '{candidate}' for category '{clean_cat}'")
                    return {
                        "status": "RESOLVED",
                        "category": clean_cat,
                        "sound_cue": sound_cue,
                        "asset_path": str(cand_path.resolve()),
                        "asset_name": candidate,
                    }

        logger.info(f"AssetResolver | No local asset found for SFX cue: '{sound_cue}' ({clean_cat}) -> UNRESOLVED_ASSET")
        return {
            "status": "UNRESOLVED_ASSET",
            "category": clean_cat,
            "sound_cue": sound_cue,
            "asset_path": None,
            "asset_name": None,
        }
