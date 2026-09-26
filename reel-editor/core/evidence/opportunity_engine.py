"""
core/evidence/opportunity_engine.py

Editing Opportunity Inventory Engine (Phase 2)
─────────────────────────────────────────────
Extracts candidate editing opportunities from the Evidence Inventory and Story Observations.

Key Design Rules:
  - Agent 1 is an OBSERVER, not an editor.
  - Every opportunity MUST reference underlying evidence IDs.
  - NO creative styling or imperative edit specs (no zoom values like 1.1x/1.25x, no colors, no font names).
  - Traceability: transcript word → observation → evidence → opportunity.
"""

from __future__ import annotations

from typing import Any, Optional

from core.logger import get_logger
from core.evidence.models import (
    EvidenceInventory,
    StoryObservationsInventory,
    EditOpportunity,
    OpportunityInventory,
    OPPORTUNITY_TYPES,
)

logger = get_logger(__name__)


class OpportunityObserver:
    """Generates candidate edit opportunities strictly grounded in multi-modal evidence."""

    def __init__(
        self,
        evidence_inv: EvidenceInventory,
        story_obs: StoryObservationsInventory,
        duration: float,
    ):
        self.evidence_inv = evidence_inv
        self.story_obs = story_obs
        self.duration = duration
        self._opp_counter = 0

    def _next_id(self) -> str:
        self._opp_counter += 1
        return f"opp_{self._opp_counter:03d}"

    def build_opportunity_inventory(self) -> OpportunityInventory:
        """Constructs an inventory of potential edit opportunities for downstream review."""
        self._opp_counter = 0
        opportunities: list[EditOpportunity] = []

        # ── 1. Dead Space Cut Opportunities ────────────────────────────
        dead_spaces = self.evidence_inv.filter_by_type("dead_space")
        for ds in dead_spaces:
            dur = ds.end - ds.start
            prio = min(0.95, round(0.5 + dur * 0.4, 2))
            opportunities.append(
                EditOpportunity(
                    opportunity_id=self._next_id(),
                    type="cut_opportunity",
                    time_range=[ds.start, ds.end],
                    evidence_ids=[ds.evidence_id],
                    confidence=ds.confidence,
                    priority=prio,
                    description=f"Dead space interval ({dur:.2f}s) where removing hesitation may tighten overall delivery cadence.",
                )
            )

        # ── 2. Punch-In / Reframing Opportunities ─────────────────────
        punch_evs = self.evidence_inv.filter_by_type("punch_in_opportunity")
        for pev in punch_evs:
            opportunities.append(
                EditOpportunity(
                    opportunity_id=self._next_id(),
                    type="punch_in_opportunity",
                    time_range=[pev.start, pev.end],
                    evidence_ids=[pev.evidence_id],
                    confidence=pev.confidence,
                    priority=0.82,
                    description="Vocal and semantic emphasis aligned with stable facial framing; candidate for framing emphasis.",
                )
            )

        # ── 3. Acoustic Accent (SFX) Opportunities ─────────────────────
        sfx_evs = self.evidence_inv.filter_by_type("sfx_opportunity")
        for sev in sfx_evs:
            opportunities.append(
                EditOpportunity(
                    opportunity_id=self._next_id(),
                    type="sfx_opportunity",
                    time_range=[sev.start, sev.end],
                    evidence_ids=[sev.evidence_id],
                    confidence=sev.confidence,
                    priority=0.75,
                    description="Sharp vocal articulation inflection; candidate for auditory accent to reinforce point.",
                )
            )

        # ── 4. B-Roll Opportunities ────────────────────────────────────
        broll_evs = self.evidence_inv.filter_by_type("broll_opportunity")
        for bev in broll_evs:
            concepts = bev.signals.speech.get("concepts", [])
            concept_str = f" involving ({', '.join(concepts)})" if concepts else ""
            opportunities.append(
                EditOpportunity(
                    opportunity_id=self._next_id(),
                    type="broll_opportunity",
                    time_range=[bev.start, bev.end],
                    evidence_ids=[bev.evidence_id],
                    confidence=bev.confidence,
                    priority=0.70,
                    description=f"Conceptual narrative passage{concept_str}; candidate for illustrative visual cutaway.",
                )
            )

        # ── 5. Emphasis Text Opportunities ─────────────────────────────
        emphasis_evs = self.evidence_inv.filter_by_type("emphasis")
        for eev in emphasis_evs:
            word_kw = eev.signals.speech.get("word", "")
            opportunities.append(
                EditOpportunity(
                    opportunity_id=self._next_id(),
                    type="emphasis_text_opportunity",
                    time_range=[eev.start, eev.end],
                    evidence_ids=[eev.evidence_id],
                    confidence=eev.confidence,
                    priority=0.85,
                    description=f"Acoustic emphasis peak on word '{word_kw}'; candidate for textual visual emphasis.",
                )
            )

        # ── 6. Hero Text Opportunities (from Hook & Realization Acts) ──
        for so in self.story_obs.story_observations:
            if so.type in ("HOOK", "REALIZATION"):
                opportunities.append(
                    EditOpportunity(
                        opportunity_id=self._next_id(),
                        type="hero_text_opportunity",
                        time_range=[so.time_range[0], min(self.duration, so.time_range[0] + 3.0)],
                        evidence_ids=list(so.evidence_ids[:3]),
                        confidence=so.confidence,
                        priority=0.90,
                        description=f"Core thematic statement in {so.type} act; candidate for prominent titular text.",
                    )
                )

        # ── 7. NO_EDIT Opportunities (First Class) ─────────────────────
        no_edit_evs = self.evidence_inv.filter_by_type("no_edit_region")
        for nev in no_edit_evs:
            opportunities.append(
                EditOpportunity(
                    opportunity_id=self._next_id(),
                    type="no_edit_opportunity",
                    time_range=[nev.start, nev.end],
                    evidence_ids=[nev.evidence_id],
                    confidence=nev.confidence,
                    priority=0.95,
                    description="Continuous natural pacing and stable visual delivery; recommended for preservation without intervention.",
                )
            )

        # Sort opportunities chronologically
        opportunities.sort(key=lambda o: (o.time_range[0], o.time_range[1], o.opportunity_id))

        return OpportunityInventory(
            version="3.0.0",
            source_video=self.evidence_inv.source_video,
            duration_seconds=self.duration,
            opportunities=opportunities,
            metadata={
                "total_opportunities": len(opportunities),
                "types_count": {
                    ot: sum(1 for o in opportunities if o.type == ot)
                    for ot in set(o.type for o in opportunities)
                },
            },
        )
