"""
agents/agent3_creative/__init__.py

Agent 3: Creative Director Package
──────────────────────────────────
Public exports for Agent 3 (Creative Decision Layer & Timeline IR Compilation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

from agents.agent3_creative.styles import (
    StyleProfile, StyleProfileName, get_style_profile
)
from agents.agent3_creative.models import (
    CreativeDecision, CreativePlan, CreativeOperation, CreativeStatus,
    TypographySpec, AnimationSpec, PunchInSpec, BrollSpec, SfxSpec, AudioSpec
)
from agents.agent3_creative.assets import (
    BaseAssetResolver, LocalAssetLibraryResolver
)
from agents.agent3_creative.creative_director import CreativeDirector
from agents.agent3_creative.timeline_compiler import CreativeTimelineCompiler
from agents.agent3_creative.reporter import CreativeReporter
from agents.agent3_creative.validator import CreativeDirectorValidator
from core.json_utils import save_json
from core.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "StyleProfile",
    "StyleProfileName",
    "get_style_profile",
    "CreativeDecision",
    "CreativePlan",
    "CreativeOperation",
    "CreativeStatus",
    "TypographySpec",
    "AnimationSpec",
    "PunchInSpec",
    "BrollSpec",
    "SfxSpec",
    "AudioSpec",
    "BaseAssetResolver",
    "LocalAssetLibraryResolver",
    "CreativeDirector",
    "CreativeTimelineCompiler",
    "CreativeReporter",
    "CreativeDirectorValidator",
    "run",
]


def run(
    output_dir: Path | str,
    source_video_path: Path | str,
    style_profile_name: str = "EDITORIAL_CINEMATIC",
    edit_decisions: Optional[Dict[str, Any]] = None,
    story_edit_plan: Optional[Dict[str, Any]] = None,
    media_meta: Optional[Dict[str, Any]] = None,
    transcript_raw: Optional[Dict[str, Any]] = None,
    visual_features: Optional[Dict[str, Any]] = None,
    fused_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes Agent 3 (Creative Director):
    1. Directs creative decisions from Agent 2's approved edit decisions.
    2. Compiles the CreativePlan into the authoritative multi-track TimelineIR.
    3. Generates the interactive HTML creative report.
    4. Validates all 13 acceptance checks.
    """
    out_dir = Path(output_dir).resolve()
    vid_path = Path(source_video_path).resolve()

    # Load artifacts if not passed directly
    if edit_decisions is None:
        p = out_dir / "edit_decisions.json"
        if p.exists():
            edit_decisions = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"Missing required artifact: {p}")

    if story_edit_plan is None:
        p = out_dir / "story_edit_plan.json"
        if p.exists():
            story_edit_plan = json.loads(p.read_text(encoding="utf-8"))

    if media_meta is None:
        p = out_dir / "media_meta.json"
        if p.exists():
            media_meta = json.loads(p.read_text(encoding="utf-8"))
        else:
            media_meta = {"width": 2160, "height": 1192, "duration_seconds": 90.833, "fps": 30.0}

    if transcript_raw is None:
        p = out_dir / "enriched_transcript.json"
        if not p.exists():
            p = out_dir / "transcript_raw.json"
        if p.exists():
            transcript_raw = json.loads(p.read_text(encoding="utf-8"))

    if visual_features is None:
        p = out_dir / "visual_features.json"
        if p.exists():
            visual_features = json.loads(p.read_text(encoding="utf-8"))

    # 1. Creative Directing
    logger.info(f"Agent 3 | Initializing Creative Director (Profile: {style_profile_name})...")
    director = CreativeDirector(
        style_profile=style_profile_name,
        workspace_root=out_dir.parent.parent,
    )
    creative_plan = director.direct(
        edit_decisions_data=edit_decisions,
        story_edit_plan_data=story_edit_plan,
        media_meta=media_meta,
        visual_features=visual_features,
        transcript_raw=transcript_raw,
    )

    # Save creative_decisions.json
    creative_decisions_path = out_dir / "creative_decisions.json"
    save_json(creative_plan.to_dict(), creative_decisions_path)
    logger.info(f"Agent 3 | Saved creative decisions to '{creative_decisions_path.name}'.")

    # 2. Compile Authoritative Timeline IR
    logger.info("Agent 3 | Compiling Creative Plan into authoritative Timeline IR...")
    compiler = CreativeTimelineCompiler()
    timeline_ir, reframe_plan = compiler.compile(
        creative_plan=creative_plan,
        source_video_path=vid_path,
        media_meta=media_meta,
        transcript_raw=transcript_raw,
        fused_timeline=fused_timeline,
    )

    timeline_ir_path = out_dir / "timeline_ir.json"
    save_json(timeline_ir.to_dict(), timeline_ir_path)
    logger.info(f"Agent 3 | Saved authoritative Timeline IR v{timeline_ir.version} to '{timeline_ir_path.name}'.")

    # 3. Generate HTML Report
    reporter = CreativeReporter()
    report_path = out_dir / "creative_report.html"
    reporter.generate(creative_plan, edit_decisions, report_path)
    logger.info(f"Agent 3 | Saved creative report to '{report_path.name}'.")

    # 4. Acceptance Validation
    validator = CreativeDirectorValidator()
    val_report = validator.validate(
        creative_plan=creative_plan,
        edit_decisions=edit_decisions,
        timeline_ir=timeline_ir,
    )
    val_path = out_dir / "creative_validation_report.json"
    save_json(val_report, val_path)

    if not val_report["passed"]:
        logger.warning(f"Agent 3 | Validation reported issues: {val_report['errors']}")
    else:
        logger.info(f"Agent 3 | All {val_report['total_checks']} acceptance checks PASSED.")

    return {
        "status": "success" if val_report["passed"] else "validation_warnings",
        "creative_plan": creative_plan.to_dict(),
        "timeline_ir": timeline_ir.to_dict(),
        "creative_decisions_file": str(creative_decisions_path),
        "timeline_ir_file": str(timeline_ir_path),
        "creative_report_file": str(report_path),
        "validation_report_file": str(val_path),
        "validation_passed": val_report["passed"],
    }
