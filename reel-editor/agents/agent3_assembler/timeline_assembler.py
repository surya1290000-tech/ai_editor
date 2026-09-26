"""
agents/agent3_assembler/timeline_assembler.py

Agent 3: Multi-Track Timeline Assembler, Render Compiler & Edit Validator
─────────────────────────────────────────────────────────────────────────
Role:
1. Composes the authoritative multi-track Timeline IR from Agent 2's creative decisions.
2. Compiles the Timeline IR into FFmpeg execution steps (face-aware 9:16 reframe, jump-cut punch-ins,
   burned-in kinetic ASS captions, DaVinci Resolve CMX 3600 EDL).
3. Automatically executes forensic Edit Validation, verifying that every expected edit operation
   was physically rendered into the final delivered MP4.
4. Fails execution if expected edits are missing.

Cost: ₹0 local execution.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Any, Dict

import core  # Ensures Gyan FFmpeg on PATH
from core.logger import get_logger
from core.json_utils import save_json
from core.timeline import (
    TimelineComposer, RenderCompiler, EditValidator, TimelineIR
)

logger = get_logger(__name__)


def run(
    output_dir: Path | str,
    video_path: Path | str,
    creative_decisions: Optional[dict] = None,
    fused_timeline: Optional[dict] = None,
    transcript_raw: Optional[dict] = None,
    media_meta: Optional[dict] = None,
    visual_features: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Run Agent 3: Assemble Timeline IR, render final MP4 with RenderCompiler,
    and forensically validate the render with EditValidator.
    """
    output_dir = Path(output_dir).resolve()
    video_path = Path(video_path).resolve()

    # Load required artifacts if not directly supplied
    if creative_decisions is None:
        with open(output_dir / "creative_decisions.json", "r", encoding="utf-8") as f:
            creative_decisions = json.load(f)
    if fused_timeline is None:
        with open(output_dir / "fused_timeline.json", "r", encoding="utf-8") as f:
            fused_timeline = json.load(f)
    if transcript_raw is None:
        p = output_dir / "transcript_raw.json"
        transcript_raw = json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}
    if media_meta is None:
        p = output_dir / "media_meta.json"
        media_meta = json.load(open(p, "r", encoding="utf-8")) if p.exists() else {}
    if visual_features is None:
        p = output_dir / "visual_features.json"
        visual_features = json.load(open(p, "r", encoding="utf-8")) if p.exists() else None

    logger.info(f"Agent 3 | Initializing Timeline Composer for '{video_path.name}'...")
    t0 = time.time()

    # ── 1. Compose Authoritative Multi-Track Timeline IR ───────────
    composer = TimelineComposer()
    timeline_ir, reframe_plan = composer.compose(
        source_video_path=video_path,
        media_meta=media_meta,
        creative_decisions=creative_decisions,
        transcript_raw=transcript_raw,
        fused_timeline=fused_timeline,
        visual_features=visual_features,
    )

    timeline_ir_path = output_dir / "timeline_ir.json"
    save_json(timeline_ir.to_dict(), timeline_ir_path)
    logger.info(
        f"Agent 3 | Saved Timeline IR v{timeline_ir.version} to '{timeline_ir_path.name}' "
        f"({timeline_ir.timeline_duration:.2f}s, {len(timeline_ir.video_tracks[0].clips)} clips)"
    )

    # ── 2. Compile Timeline IR to Rendered MP4 & DaVinci EDL ─────────
    compiler = RenderCompiler()
    render_stats = compiler.compile(
        timeline=timeline_ir,
        output_dir=output_dir,
        burn_captions=creative_decisions.get("creative_style", {}).get("burn_subtitles", True),
        output_filename="final_reel_edited.mp4",
    )

    final_video = render_stats["final_video"]
    ass_file = render_stats.get("ass_file")
    srt_file = render_stats.get("srt_file")
    edl_file = render_stats.get("edl_file")

    # ── 3. Forensic Edit Validation ─────────────────────────────────
    validator = EditValidator()
    validation_report = validator.validate(
        final_video_path=final_video,
        timeline=timeline_ir,
        ass_path=ass_file,
    )

    val_report_path = output_dir / "validation_report.json"
    save_json(validation_report.to_dict(), val_report_path)

    if not validation_report.is_valid:
        error_msg = f"Agent 3 | Edit Validation FAILED: {'; '.join(validation_report.errors)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # ── 4. Build Output Summary ────────────────────────────────────
    total_time = round(time.time() - t0, 2)
    summary = {
        "status": "success",
        "final_video": final_video,
        "video_filename": "final_reel_edited.mp4",
        "file_size_mb": render_stats.get("file_size_mb", 0.0),
        "render_time_seconds": total_time,
        "timeline_ir_file": str(timeline_ir_path),
        "validation_report_file": str(val_report_path),
        "edl_file": edl_file,
        "srt_file": srt_file,
        "ass_file": ass_file,
        "rendered_duration": validation_report.measured_duration,
        "resolution": validation_report.measured_resolution,
        "aspect_ratio": validation_report.aspect_ratio,
        "cuts_executed": validation_report.cuts_executed,
        "punch_ins_executed": validation_report.punch_ins_executed,
        "captions_compiled": validation_report.captions_compiled,
        "validation_status": validation_report.status,
    }

    save_json(summary, output_dir / "render_summary.json")
    logger.info(
        f"Agent 3 | Render & Validation Succeeded! '{Path(final_video).name}' "
        f"({validation_report.measured_resolution}, {validation_report.measured_duration:.1f}s, "
        f"{validation_report.punch_ins_executed} punch-ins) in {total_time}s."
    )

    return summary
