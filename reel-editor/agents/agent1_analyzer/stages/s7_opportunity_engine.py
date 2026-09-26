"""
agents/agent1_analyzer/stages/s7_opportunity_engine.py

Stage 7: Evidence Synthesis & Opportunity Engine (Phase 2 Standardized)
───────────────────────────────────────────────────────────────────────
INPUT:  media_meta.json          (Stage 1)
        transcript_raw.json      (Stage 2)
        audio_features.json      (Stage 3)
        visual_features.json     (Stage 4)
        fused_timeline.json      (Stage 5)
        story_analysis.json      (Stage 6)

OUTPUT: enriched_transcript.json
        evidence_inventory.json
        story_observations.json
        opportunity_inventory.json
        analysis_report.html

Design principles (Phase 2):
  - Agent 1 is strictly an OBSERVER answering "What is happening in this footage?"
  - NO creative decisions (no font choices, no zoom values, no styling, no SFX tracks).
  - First-class NO_EDIT regions.
  - Low-confidence ASR words propagate uncertainty to downstream semantics.
  - Every opportunity is strictly grounded in evidence IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

from core.logger import get_logger
from core.evidence.pipeline import run_agent1_observer_pipeline

logger = get_logger(__name__)


def run(
    output_dir: Path | str,
    fused_timeline: Optional[dict] = None,
    story_analysis: Optional[dict] = None,
    media_meta: Optional[dict] = None,
    transcript_raw: Optional[dict] = None,
    audio_features: Optional[dict] = None,
    visual_features: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Run Stage 7: Evidence Synthesis & Opportunity Engine.
    Executes the Phase 2 standardized Observer evidence pipeline.
    """
    output_dir = Path(output_dir)
    logger.info("Stage 7 | Running Phase 2 Standardized Observer Pipeline...")

    results = run_agent1_observer_pipeline(
        output_dir=output_dir,
        media_meta=media_meta,
        transcript_raw=transcript_raw,
        audio_features=audio_features,
        visual_features=visual_features,
        fused_timeline=fused_timeline,
        story_analysis=story_analysis,
    )

    opp_inv = results["opportunity_inventory"]
    return opp_inv.to_dict()
