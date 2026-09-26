"""
scripts/run_phase6_human_review.py

Master Runner for Phase 6: Human-Quality Editorial Review & Editable Timeline
─────────────────────────────────────────────────────────────────────────────
Executes:
1. EditorialReviewer:
   - Captures raw vs edited side-by-side frames for all 9 key moments
   - Scores 5 quality dimensions (story, visual, pacing, distraction, naturalness)
   - Computes edit density metrics & classifies pacing
   - Evaluates all 10 in-depth editorial dimensions
2. EditorialQualityReporter:
   - Generates interactive, publication-grade human_quality_report.html
3. TimelineProjectManager:
   - Preserves timeline_ir_v1.json as immutable AI baseline
   - Initializes manifest.json for version branching and rollback
   - Verifies traceability lookup and export functions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core
from core.logger import get_logger
from core.review.reviewer import EditorialReviewer
from core.review.reporter import EditorialQualityReporter
from core.timeline.project import TimelineProjectManager

logger = get_logger("phase6_runner")


def main():
    logger.info("==================================================")
    logger.info("STARTING PHASE 6: HUMAN-QUALITY REVIEW & EDITABLE EDITOR")
    logger.info("==================================================")

    # Resolve paths
    raw_video = PROJECT_ROOT / "Surya.mp4"
    if not raw_video.exists():
        # Look in inputs or parent
        candidates = list(PROJECT_ROOT.glob("**/Surya.mp4"))
        if candidates:
            raw_video = candidates[0]

    output_dir = PROJECT_ROOT / "outputs" / "Surya_phase2_observer"
    edited_video = output_dir / "final_reel_edited.mp4"
    timeline_ir_file = output_dir / "timeline_ir.json"
    edit_decisions_file = output_dir / "edit_decisions.json"
    creative_decisions_file = output_dir / "creative_decisions.json"

    if not edited_video.exists():
        logger.error(f"Rendered video not found at {edited_video}. Please complete Phase 5 render first.")
        sys.exit(1)

    with open(timeline_ir_file, "r", encoding="utf-8") as f:
        timeline_ir = json.load(f)

    with open(edit_decisions_file, "r", encoding="utf-8") as f:
        edit_decisions = json.load(f)

    with open(creative_decisions_file, "r", encoding="utf-8") as f:
        creative_plan = json.load(f)

    # ── 1. Run Editorial Reviewer ───────────────────────────────────
    logger.info(f"Extracting frame comparisons and assessing quality rubrics...")
    reviewer = EditorialReviewer()
    review_data = reviewer.review(
        raw_video_path=raw_video,
        edited_video_path=edited_video,
        edit_decisions=edit_decisions,
        creative_plan=creative_plan,
        timeline_ir=timeline_ir,
        output_dir=output_dir,
    )

    review_dir = output_dir / "human_quality_review"
    logger.info(f"Frame extraction and scoring complete. Data saved to {review_dir / 'human_quality_data.json'}")

    # ── 2. Generate Interactive HTML Quality Report ─────────────────
    logger.info("Generating publication-grade human_quality_report.html...")
    reporter = EditorialQualityReporter()
    html_report_path = review_dir / "human_quality_report.html"
    reporter.generate_html_report(
        report_data=review_data,
        output_file=html_report_path,
        embed_images=True,
    )

    # ── 3. Initialize Versioned Project & Baseline v1 ───────────────
    logger.info("Initializing TimelineProjectManager with immutable baseline v1...")
    project_dir = output_dir / "project"
    proj_mgr = TimelineProjectManager(
        project_dir=project_dir,
        base_timeline_path=timeline_ir_file,
        context_dir=output_dir,
    )

    manifest = proj_mgr.get_manifest()
    logger.info(f"Project initialized. Active version: {manifest['current_version']}")

    # Verify decision trace on sample event (hero text)
    sample_trace = proj_mgr.build_traceability_graph("hero_title_01")
    logger.info(f"Verified decision traceability graph: target={sample_trace['target_id']}, story_op={sample_trace.get('story_decision', {}).get('operation') if sample_trace.get('story_decision') else 'N/A'}")

    # Generate test exports (EDL and SRT)
    edl_text = proj_mgr.export_edl()
    srt_text = proj_mgr.export_srt()
    logger.info(f"Verified NLE exports: EDL ({len(edl_text.splitlines())} lines), SRT ({len(srt_text.splitlines())} lines)")

    # ── Print Summary ───────────────────────────────────────────────
    density = review_data["density_metrics"]
    scores = review_data["quality_scores"]
    print("\n" + "=" * 65)
    print("PHASE 6 HUMAN-QUALITY REVIEW SUMMARY")
    print("=" * 65)
    print(f"Source Media:          {review_data['source_video']}")
    print(f"Edited Video:          {review_data['edited_video']}")
    print(f"Pacing Classification: {density['pacing_classification']}")
    print(f"Major Edits / Min:     {density['major_edits_per_minute']} / min")
    print(f"Total Edits / Min:     {density['edits_per_minute']} / min")
    print(f"SFX Density:           {density['sfx_density_per_minute']} / min")
    print(f"B-Roll Density:        {density['broll_density_per_minute']} / min")
    print("-" * 65)
    print("QUALITY RUBRIC AVERAGES (0.00 - 1.00):")
    print(f"  • Story Value:        {scores['overall_story_value']:.2f}")
    print(f"  • Visual Value:       {scores['overall_visual_value']:.2f}")
    print(f"  • Pacing Value:       {scores['overall_pacing_value']:.2f}")
    print(f"  • Distraction Cost:   {scores['overall_distraction_cost']:.2f} (lower is cleaner)")
    print(f"  • Naturalness Score:  {scores['overall_naturalness_score']:.2f}")
    print(f"  • Conceptual Verdicts: KEEP: {scores['verdicts']['keep']} | REVISE: {scores['verdicts']['revise']} | REMOVE: {scores['verdicts']['remove']}")
    print("-" * 65)
    print("DELIVERABLES GENERATED:")
    print(f"  1. HTML Review Report: {html_report_path}")
    print(f"  2. Review Data JSON:   {review_dir / 'human_quality_data.json'}")
    print(f"  3. Immutable v1 IR:    {project_dir / 'timeline_ir_v1.json'}")
    print(f"  4. Timeline Manifest:  {project_dir / 'manifest.json'}")
    print(f"  5. Web Studio URL:     http://localhost:8000/editor")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
