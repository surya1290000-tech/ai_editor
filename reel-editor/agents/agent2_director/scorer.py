"""
agents/agent2_director/scorer.py

Editorial Scoring, Budget Enforcement & Conflict Resolution (Phase 3)
──────────────────────────────────────────────────────────────────────
Implements:
  1. EditBudgetTracker: Tracks hard limits, spacing intervals, and density caps.
  2. EditorialScorer: Evaluates editorial benefit vs disruption cost.
  3. ConflictResolver: Resolves overlapping interventions based on priority hierarchy.

Priority Hierarchy:
  1. Story clarity
  2. Speech intelligibility
  3. Major narrative moment (Hook / Thesis Realization)
  4. Emphasis (vocal/semantic peak)
  5. Framing (punch-in)
  6. B-roll (conceptual context)
  7. SFX (auditory punctuation)
  8. Decorative effects
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)


# Default fallback budget configuration if yaml missing
DEFAULT_BUDGET_CONFIG = {
    "edit_budget": {
        "hero_text_max": 2,
        "editorial_card_max": 1,
        "strong_punch_in_max": 3,
        "medium_punch_in_max": 5,
        "subtle_punch_in_max": 8,
        "sfx_max": 6,
        "broll_max": 3,
        "emphasis_text_max": 8,
        "min_seconds_between_punch_ins": 4.0,
        "min_seconds_between_sfx": 3.0,
        "min_seconds_between_hero_text": 15.0,
        "min_seconds_between_broll": 8.0,
        "min_seconds_between_emphasis_text": 2.0,
        "max_edit_density_per_10s": 3,
        "no_edit_is_valid": True,
        "prefer_restraint": True,
    }
}


class EditBudgetTracker:
    """Enforces edit budgets, minimum spacing, and density ceilings."""

    def __init__(self, config_path: Optional[Path | str] = None, duration: float = 60.0):
        self.duration = duration
        self.limits: dict[str, int] = {}
        self.spacing: dict[str, float] = {}
        self.max_density_per_10s: int = 3
        self.prefer_restraint: bool = True

        # Usage tracking
        self.counts: dict[str, int] = {
            "hero_text": 0,
            "editorial_card": 0,
            "strong_punch_in": 0,
            "medium_punch_in": 0,
            "subtle_punch_in": 0,
            "sfx": 0,
            "broll": 0,
            "emphasis_text": 0,
            "cuts": 0,
            "no_edit": 0,
        }

        # Last occurrence timestamps
        self.last_timestamps: dict[str, float] = {
            "punch_in": -999.0,
            "sfx": -999.0,
            "hero_text": -999.0,
            "broll": -999.0,
            "emphasis_text": -999.0,
        }

        # Window tracking for 10s density
        self.accepted_intervention_times: list[float] = []

        self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path | str]):
        cfg = DEFAULT_BUDGET_CONFIG
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or DEFAULT_BUDGET_CONFIG
            except Exception as e:
                logger.warning(f"Could not load {config_path} ({e}), using default budget.")

        eb = cfg.get("edit_budget", {})

        # Short Video Rules (< 15 seconds): Avoid overcrowding short reels
        if self.duration < 15.0:
            self.limits = {
                "hero_text": min(1, eb.get("hero_text_max", 2)),
                "editorial_card": 0,
                "strong_punch_in": 0,
                "medium_punch_in": min(1, eb.get("medium_punch_in_max", 5)),
                "subtle_punch_in": min(1, eb.get("subtle_punch_in_max", 8)),
                "sfx": min(1, eb.get("sfx_max", 6)),
                "broll": 0,
                "emphasis_text": min(2, eb.get("emphasis_text_max", 8)),
            }
            self.spacing = {
                "punch_in": 3.0,
                "sfx": 2.5,
                "hero_text": 10.0,
                "broll": 6.0,
                "emphasis_text": 1.5,
            }
            self.max_density_per_10s = 2
            self.prefer_restraint = True
        else:
            self.limits = {
                "hero_text": eb.get("hero_text_max", 2),
                "editorial_card": eb.get("editorial_card_max", 1),
                "strong_punch_in": eb.get("strong_punch_in_max", 3),
                "medium_punch_in": eb.get("medium_punch_in_max", 5),
                "subtle_punch_in": eb.get("subtle_punch_in_max", 8),
                "sfx": eb.get("sfx_max", 6),
                "broll": eb.get("broll_max", 3),
                "emphasis_text": eb.get("emphasis_text_max", 8),
            }
            self.spacing = {
                "punch_in": eb.get("min_seconds_between_punch_ins", 4.0),
                "sfx": eb.get("min_seconds_between_sfx", 3.0),
                "hero_text": eb.get("min_seconds_between_hero_text", 15.0),
                "broll": eb.get("min_seconds_between_broll", 8.0),
                "emphasis_text": eb.get("min_seconds_between_emphasis_text", 2.0),
            }
            self.max_density_per_10s = eb.get("max_edit_density_per_10s", 3)
            self.prefer_restraint = eb.get("prefer_restraint", True)

    def can_accept(
        self,
        operation: str,
        start: float,
        end: float,
        strength: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Checks whether an operation can be accepted within budget, spacing, and density."""
        if operation in ("NO_EDIT", "KEEP"):
            return True, "No-edit / preservation is always permitted."

        # Map operation to category
        category = self._map_category(operation, strength)

        # 1. Check Hard Limit
        if category in self.limits:
            curr_count = self.counts.get(category, 0)
            max_limit = self.limits[category]
            if curr_count >= max_limit:
                return False, f"Budget limit reached ({curr_count}/{max_limit} for {category})."

        # 2. Check Spacing Interval
        spacing_key = self._map_spacing_key(operation)
        if spacing_key and spacing_key in self.spacing:
            min_space = self.spacing[spacing_key]
            last_t = self.last_timestamps.get(spacing_key, -999.0)
            if start - last_t < min_space:
                return False, f"Minimum spacing ({min_space}s) for {spacing_key} violated (only {start - last_t:.2f}s since last)."

        # 3. Check 10s Window Density
        # Major interventions: punch-in, hero text, b-roll, sfx
        if operation in ("PUNCH_IN", "HERO_TEXT", "BROLL_SUPPORT", "BROLL_CONTEXT", "BROLL_CONTRAST", "SFX_IMPACT", "SFX_WHOOSH"):
            window_interventions = [
                t for t in self.accepted_intervention_times
                if start - 10.0 <= t <= start + 1.0
            ]
            if len(window_interventions) >= self.max_density_per_10s:
                return False, f"10s density ceiling ({self.max_density_per_10s}) exceeded (already {len(window_interventions)} edits in window)."

        return True, "Within budget and spacing constraints."

    def consume(
        self,
        operation: str,
        start: float,
        end: float,
        strength: Optional[str] = None,
    ):
        """Records the consumption of budget and updates timing trackers."""
        category = self._map_category(operation, strength)
        if category in self.counts:
            self.counts[category] += 1

        spacing_key = self._map_spacing_key(operation)
        if spacing_key in self.last_timestamps:
            self.last_timestamps[spacing_key] = end

        if operation in ("PUNCH_IN", "HERO_TEXT", "BROLL_SUPPORT", "BROLL_CONTEXT", "BROLL_CONTRAST", "SFX_IMPACT", "SFX_WHOOSH"):
            self.accepted_intervention_times.append(start)

    def get_utilization_report(self) -> dict[str, Any]:
        """Returns structured budget utilization metrics."""
        return {
            "hero_text": {"used": self.counts.get("hero_text", 0), "limit": self.limits.get("hero_text", 2)},
            "punch_ins": {
                "strong": {"used": self.counts.get("strong_punch_in", 0), "limit": self.limits.get("strong_punch_in", 3)},
                "medium": {"used": self.counts.get("medium_punch_in", 0), "limit": self.limits.get("medium_punch_in", 5)},
                "subtle": {"used": self.counts.get("subtle_punch_in", 0), "limit": self.limits.get("subtle_punch_in", 8)},
                "total_used": (
                    self.counts.get("strong_punch_in", 0)
                    + self.counts.get("medium_punch_in", 0)
                    + self.counts.get("subtle_punch_in", 0)
                ),
            },
            "sfx": {"used": self.counts.get("sfx", 0), "limit": self.limits.get("sfx", 6)},
            "broll": {"used": self.counts.get("broll", 0), "limit": self.limits.get("broll", 3)},
            "emphasis_text": {"used": self.counts.get("emphasis_text", 0), "limit": self.limits.get("emphasis_text", 8)},
            "cuts": {"used": self.counts.get("cuts", 0)},
            "no_edit_regions": {"used": self.counts.get("no_edit", 0)},
        }

    def _map_category(self, operation: str, strength: Optional[str]) -> str:
        if operation == "HERO_TEXT":
            return "hero_text"
        if operation == "EDITORIAL_CARD":
            return "editorial_card"
        if operation == "PUNCH_IN":
            if strength == "STRONG":
                return "strong_punch_in"
            if strength == "SUBTLE":
                return "subtle_punch_in"
            return "medium_punch_in"
        if operation in ("SFX_WHOOSH", "SFX_POP", "SFX_IMPACT", "SFX_RISER"):
            return "sfx"
        if operation in ("BROLL_SUPPORT", "BROLL_CONTEXT", "BROLL_CONTRAST"):
            return "broll"
        if operation == "EMPHASIS_TEXT":
            return "emphasis_text"
        if operation in ("CUT", "REMOVE_DEAD_SPACE", "REMOVE_FILLER", "PACE_TRIM", "SOFT_CUT"):
            return "cuts"
        if operation in ("NO_EDIT", "KEEP"):
            return "no_edit"
        return "other"

    def _map_spacing_key(self, operation: str) -> Optional[str]:
        if operation == "PUNCH_IN":
            return "punch_in"
        if operation in ("SFX_WHOOSH", "SFX_POP", "SFX_IMPACT", "SFX_RISER"):
            return "sfx"
        if operation == "HERO_TEXT":
            return "hero_text"
        if operation in ("BROLL_SUPPORT", "BROLL_CONTEXT", "BROLL_CONTRAST"):
            return "broll"
        if operation == "EMPHASIS_TEXT":
            return "emphasis_text"
        return None


# ─── Editorial Scorer ─────────────────────────────────────────────────────────

class EditorialScorer:
    """Evaluates the net editorial value of candidate editing actions."""

    def score_candidate(
        self,
        operation: str,
        story_act: str,
        moment_classification: str,
        importance: float,
        confidence: float,
        has_uncertainty: bool,
        duration: float,
    ) -> float:
        """
        Computes composite score:
          Value = clarity + pacing + emphasis + narrative + retention - disruption
        """
        # Baseline benefits
        narrative_weight = 1.2 if story_act in ("HOOK", "REALIZATION", "PAYOFF") else 0.9
        moment_mult = 1.3 if moment_classification == "PRIMARY" else 0.9 if moment_classification == "SECONDARY" else 0.4

        # In PRESERVE regions, artificial interventions have severe disruption penalty
        if moment_classification == "PRESERVE" and operation not in ("NO_EDIT", "KEEP"):
            return -0.5

        if operation in ("NO_EDIT", "KEEP"):
            # High value if in natural delivery
            return 0.85 * moment_mult

        if operation in ("REMOVE_DEAD_SPACE", "CUT", "PACE_TRIM"):
            # Pacing benefit
            benefit = 0.70 + (0.2 if duration >= 0.5 else 0.05)
            penalty = 0.3 if has_uncertainty else 0.0
            return (benefit - penalty) * confidence

        if operation == "HERO_TEXT":
            # Hero text is strictly high-value only in HOOK or central REALIZATION
            if story_act in ("HOOK", "REALIZATION") and importance >= 0.80:
                benefit = 0.90 * narrative_weight
                return benefit * (0.6 if has_uncertainty else 1.0)
            return -0.2  # Reject everywhere else

        if operation == "PUNCH_IN":
            # Framing shift value
            benefit = (0.75 * importance) * narrative_weight
            penalty = 0.25 if has_uncertainty else 0.1
            return (benefit - penalty) * confidence

        if operation == "EMPHASIS_TEXT":
            # Vocal + semantic keyword reinforcement
            benefit = (0.70 * importance)
            penalty = 0.30 if has_uncertainty else 0.05
            return (benefit - penalty) * confidence

        if operation in ("BROLL_SUPPORT", "BROLL_CONTEXT", "BROLL_CONTRAST"):
            # Conceptual cutaway value
            if story_act in ("CONTEXT", "PROBLEM"):
                benefit = 0.75 * narrative_weight
                penalty = 0.2 if has_uncertainty else 0.1
                return (benefit - penalty) * confidence
            return 0.3 * confidence

        if operation in ("SFX_WHOOSH", "SFX_POP", "SFX_IMPACT", "SFX_RISER"):
            # SFX value
            benefit = 0.60 * importance
            penalty = 0.2
            return (benefit - penalty) * confidence

        return 0.5 * confidence
