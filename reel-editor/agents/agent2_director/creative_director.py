"""
agents/agent2_director/creative_director.py

Agent 2: True Story Editor Runner (Phase 3 Standardized)
───────────────────────────────────────────────────────
INPUT:  evidence_inventory.json     (Phase 2)
        story_observations.json     (Phase 2)
        opportunity_inventory.json  (Phase 2)
        enriched_transcript.json    (Phase 2)
        media_meta.json             (Stage 1)
        config/edit_budget.yaml     (Project Configuration)

OUTPUT: story_edit_plan.json
        edit_decisions.json
        editorial_report.html
        editorial_validation_report.json
        creative_decisions.json     (Backward compatibility)

Answers: "What should actually change in this reel?"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

from core.logger import get_logger
from core.json_utils import save_json
from core.evidence.models import (
    EvidenceInventory,
    StoryObservationsInventory,
    OpportunityInventory,
    EnrichedTranscript,
)
from agents.agent2_director.story_editor import StoryEditor
from agents.agent2_director.validator import StoryEditorValidator
from agents.agent2_director.reporter import generate_editorial_html_report

logger = get_logger(__name__)


def run(
    output_dir: Path | str,
    evidence_inventory: Optional[dict] = None,
    story_observations: Optional[dict] = None,
    opportunity_inventory: Optional[dict] = None,
    enriched_transcript: Optional[dict] = None,
    media_meta: Optional[dict] = None,
    pacing_preset: str = "balanced",
    budget_config_path: Optional[Path | str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Executes Agent 2: Story Editor.
    Analyzes narrative priorities, evaluates opportunities with selective restraint,
    and emits standardized Edit Decisions.
    """
    output_dir = Path(output_dir)
    logger.info(f"Agent 2 Story Editor | Starting editorial intelligence (Preset: {pacing_preset})...")

    # 1. Resolve inputs from disk if needed
    if evidence_inventory is None:
        with open(output_dir / "evidence_inventory.json", "r", encoding="utf-8") as f:
            evidence_inventory = json.load(f)
    if story_observations is None:
        with open(output_dir / "story_observations.json", "r", encoding="utf-8") as f:
            story_observations = json.load(f)
    if opportunity_inventory is None:
        with open(output_dir / "opportunity_inventory.json", "r", encoding="utf-8") as f:
            opportunity_inventory = json.load(f)
    if enriched_transcript is None:
        with open(output_dir / "enriched_transcript.json", "r", encoding="utf-8") as f:
            enriched_transcript = json.load(f)
    if media_meta is None:
        with open(output_dir / "media_meta.json", "r", encoding="utf-8") as f:
            media_meta = json.load(f)

    # 2. Deserialization into typed models
    ev_inv = EvidenceInventory.from_dict(evidence_inventory)
    st_obs = StoryObservationsInventory.from_dict(story_observations)
    opp_inv = OpportunityInventory.from_dict(opportunity_inventory)
    tx = EnrichedTranscript.from_dict(enriched_transcript)

    # Resolve budget path
    if budget_config_path is None:
        default_cfg = Path(__file__).parent.parent.parent / "config" / "edit_budget.yaml"
        if default_cfg.exists():
            budget_config_path = default_cfg

    # 3. Instantiate and run Story Editor
    editor = StoryEditor(
        evidence_inv=ev_inv,
        story_obs=st_obs,
        opp_inv=opp_inv,
        transcript=tx,
        media_meta=media_meta,
        pacing_preset=pacing_preset,
        budget_config_path=budget_config_path,
    )

    results = editor.run_editorial_pipeline()
    story_plan = results["story_edit_plan"]
    decisions_doc = results["edit_decisions"]

    # 4. Save artifacts
    save_json(story_plan.to_dict(), output_dir / "story_edit_plan.json")
    save_json(decisions_doc.to_dict(), output_dir / "edit_decisions.json")
    # Also save creative_decisions.json for backward compatibility
    save_json(decisions_doc.to_dict(), output_dir / "creative_decisions.json")

    logger.info(f"Agent 2 Story Editor | Saved story_edit_plan.json & edit_decisions.json")

    # 5. Generate Editorial HTML Report
    html_path = output_dir / "editorial_report.html"
    generate_editorial_html_report(
        story_plan=story_plan,
        decisions_doc=decisions_doc,
        transcript=tx,
        evidence_inv=ev_inv,
        output_path=html_path,
    )
    logger.info(f"Agent 2 Story Editor | Saved editorial_report.html")

    # 6. Validate All Acceptance Criteria
    validator = StoryEditorValidator(
        story_plan=story_plan,
        decisions_doc=decisions_doc,
        max_duration=float(media_meta.get("duration_seconds", 90.0)),
    )
    val_report = validator.validate_all()
    save_json(val_report, output_dir / "editorial_validation_report.json")

    if val_report["passed"]:
        logger.info("Agent 2 Story Editor | All 10 editorial acceptance checks PASSED.")
    else:
        logger.warning(f"Agent 2 Story Editor | Validation errors: {val_report['errors']}")

    return decisions_doc.to_dict()
