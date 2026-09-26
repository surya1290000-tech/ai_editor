"""
core/evidence/pipeline.py

Agent 1 Observer Orchestration Pipeline (Phase 2)
────────────────────────────────────────────────
Orchestrates the complete Observer flow:
  1. Builds Enriched Transcript (with uncertainty flags & acoustic features)
  2. Extracts Standardized Evidence Inventory (evidence_inventory.json)
  3. Derives Story Observations (story_observations.json)
  4. Catalogues Edit Opportunities (opportunity_inventory.json)
  5. Renders interactive HTML report (analysis_report.html)
  6. Validates all 10 acceptance criteria (via EvidenceLayerValidator)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from core.logger import get_logger
from core.json_utils import save_json
from core.evidence.collector import EvidenceCollector
from core.evidence.story_analyzer import StoryObserver
from core.evidence.opportunity_engine import OpportunityObserver
from core.evidence.reporter import generate_analysis_html_report
from core.evidence.validator import EvidenceLayerValidator
from core.evidence.models import (
    EnrichedTranscript,
    EvidenceInventory,
    StoryObservationsInventory,
    OpportunityInventory,
)

logger = get_logger(__name__)


def run_agent1_observer_pipeline(
    output_dir: Path | str,
    media_meta: Optional[dict[str, Any]] = None,
    transcript_raw: Optional[dict[str, Any]] = None,
    audio_features: Optional[dict[str, Any]] = None,
    visual_features: Optional[dict[str, Any]] = None,
    fused_timeline: Optional[dict[str, Any]] = None,
    story_analysis: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Executes the complete Agent 1 Observer evidence pipeline.
    Saves:
      - enriched_transcript.json
      - evidence_inventory.json
      - story_observations.json
      - opportunity_inventory.json
      - analysis_report.html
    """
    output_dir = Path(output_dir)

    # 1. Load missing inputs from disk if needed
    if media_meta is None:
        with open(output_dir / "media_meta.json", "r", encoding="utf-8") as f:
            media_meta = json.load(f)
    if transcript_raw is None:
        with open(output_dir / "transcript_raw.json", "r", encoding="utf-8") as f:
            transcript_raw = json.load(f)
    if audio_features is None:
        with open(output_dir / "audio_features.json", "r", encoding="utf-8") as f:
            audio_features = json.load(f)
    if visual_features is None:
        with open(output_dir / "visual_features.json", "r", encoding="utf-8") as f:
            visual_features = json.load(f)
    if fused_timeline is None:
        with open(output_dir / "fused_timeline.json", "r", encoding="utf-8") as f:
            fused_timeline = json.load(f)
    if story_analysis is None:
        sa_path = output_dir / "story_analysis.json"
        if sa_path.exists():
            with open(sa_path, "r", encoding="utf-8") as f:
                story_analysis = json.load(f)

    logger.info("Agent 1 Observer Pipeline | Starting evidence synthesis...")

    # 2. Build Enriched Transcript and Evidence Inventory
    collector = EvidenceCollector(
        media_meta=media_meta,
        transcript_raw=transcript_raw,
        audio_features=audio_features,
        visual_features=visual_features,
        fused_timeline=fused_timeline,
    )

    enriched_transcript = collector.build_enriched_transcript()
    save_json(enriched_transcript.to_dict(), output_dir / "enriched_transcript.json")
    logger.info(f"Agent 1 Observer Pipeline | Saved enriched_transcript.json ({len(enriched_transcript.words)} words)")

    evidence_inv = collector.collect_evidence_inventory(enriched_transcript)
    save_json(evidence_inv.to_dict(), output_dir / "evidence_inventory.json")
    logger.info(f"Agent 1 Observer Pipeline | Saved evidence_inventory.json ({len(evidence_inv.evidence)} items)")

    # 3. Derive Story Observations
    duration = float(media_meta.get("duration_seconds", 0.0))
    story_observer = StoryObserver(
        evidence_inv=evidence_inv,
        transcript=enriched_transcript,
        duration=duration,
    )
    story_obs = story_observer.analyze_story_structure(story_analysis)
    save_json(story_obs.to_dict(), output_dir / "story_observations.json")
    logger.info(f"Agent 1 Observer Pipeline | Saved story_observations.json ({len(story_obs.story_observations)} acts)")

    # 4. Catalog Edit Opportunities
    opp_observer = OpportunityObserver(
        evidence_inv=evidence_inv,
        story_obs=story_obs,
        duration=duration,
    )
    opp_inv = opp_observer.build_opportunity_inventory()
    save_json(opp_inv.to_dict(), output_dir / "opportunity_inventory.json")
    logger.info(f"Agent 1 Observer Pipeline | Saved opportunity_inventory.json ({len(opp_inv.opportunities)} opportunities)")

    # 5. Generate Visual HTML Report
    html_path = output_dir / "analysis_report.html"
    generate_analysis_html_report(
        evidence_inv=evidence_inv,
        story_obs=story_obs,
        opp_inv=opp_inv,
        transcript=enriched_transcript,
        output_path=html_path,
    )
    logger.info(f"Agent 1 Observer Pipeline | Saved analysis_report.html")

    # 6. Validate All Acceptance Criteria
    validator = EvidenceLayerValidator(
        evidence_inv=evidence_inv,
        story_obs=story_obs,
        opp_inv=opp_inv,
        transcript=enriched_transcript,
        max_duration=duration,
    )
    validation_results = validator.validate_all()
    save_json(validation_results, output_dir / "evidence_validation_report.json")

    if validation_results["passed"]:
        logger.info("Agent 1 Observer Pipeline | All 10 acceptance checks PASSED.")
    else:
        logger.warning(f"Agent 1 Observer Pipeline | Validation errors: {validation_results['errors']}")

    return {
        "enriched_transcript": enriched_transcript,
        "evidence_inventory": evidence_inv,
        "story_observations": story_obs,
        "opportunity_inventory": opp_inv,
        "html_report_path": str(html_path),
        "validation_results": validation_results,
    }
