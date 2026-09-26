"""
scripts/run_phase5_full_render.py

Phase 5: Full End-to-End Pipeline Execution on Surya.mp4
────────────────────────────────────────────────────────
Executes:
1. Agent 3 Creative Direction with fully resolved local B-roll & SFX library assets.
2. Authoritative Timeline IR compilation (v2.1.0).
3. Physical Multi-Track Render Compilation:
   - 9:16 Vertical Face-Aware Framing
   - Jump-Cut Punch-Ins (Subtle & Strong)
   - B-Roll Cutaway Video Overlays (Crossfade transitions)
   - SFX Audio Mixing (Acoustic accent without voice collision)
   - Dual-Style Subtitles (Center HeroText title cards + Kinetic captions)
   - CMX 3600 EDL for DaVinci Resolve
4. Forensic Critic / QA Pass:
   - Verifies all 10 approved editorial decisions against physical video
   - Measures audio waveforms, headroom, and clipping
   - Extracts representative frame captures
   - Emits render_validation_report.json and render_qa_report.html
"""

import sys
import json
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import core
from agents.agent3_creative import run as run_agent3
from core.timeline.compiler import RenderCompiler
from core.timeline.critic_qa import CriticQAEvaluator
from core.timeline.models import TimelineIR
from core.logger import get_logger

logger = get_logger("run_phase5_full_render")


def main():
    print("=" * 75)
    print("PHASE 5: FULL PIPELINE EXECUTION & FORENSIC CRITIC/QA PASS ON SURYA.MP4")
    print("=" * 75)

    obs_dir = REPO_ROOT / "outputs" / "Surya_phase2_observer"
    source_video = REPO_ROOT / "Surya.mp4"

    if not obs_dir.exists():
        print(f"ERROR: Observer output directory '{obs_dir}' not found.")
        sys.exit(1)
    if not source_video.exists():
        print(f"ERROR: Source video '{source_video}' not found.")
        sys.exit(1)

    t0 = time.time()

    # ── Step 1: Agent 3 Creative Direction (With Resolved Assets) ───
    print("\n[STEP 1/3] Directing Creative Decisions & Resolving Assets...")
    a3_res = run_agent3(
        output_dir=obs_dir,
        source_video_path=source_video,
        style_profile_name="EDITORIAL_CINEMATIC",
    )

    plan_dict = a3_res["creative_plan"]
    print(f"  Creative Decisions: {len(plan_dict.get('decisions', []))}")
    print(f"  Resolved Operations: {plan_dict.get('resolved_count')}")
    print(f"  Unresolved Assets: {plan_dict.get('unresolved_assets_count')}")

    # Load compiled Timeline IR
    with open(obs_dir / "timeline_ir.json", "r", encoding="utf-8") as f:
        timeline_data = json.load(f)
    timeline_ir = TimelineIR.from_dict(timeline_data)

    # ── Step 2: Multi-Track Render Compilation ──────────────────────
    print("\n[STEP 2/3] Compiling Multi-Track Timeline to final_reel_edited.mp4...")
    t_render_start = time.time()
    compiler = RenderCompiler()
    render_stats = compiler.compile(
        timeline=timeline_ir,
        output_dir=obs_dir,
        burn_captions=True,
        output_filename="final_reel_edited.mp4",
    )
    t_render = round(time.time() - t_render_start, 2)
    final_video = Path(render_stats["final_video"])
    print(f"  Render Complete in {t_render}s:")
    print(f"    Output File:   {final_video.name} ({render_stats.get('file_size_mb')} MB)")
    print(f"    Subclips:      {render_stats.get('clips_rendered')}")
    print(f"    Punch-Ins:     {render_stats.get('punch_ins_rendered')}")
    print(f"    B-Roll Overlays: {render_stats.get('broll_rendered')}")
    print(f"    SFX Cues:      {render_stats.get('sfx_rendered')}")

    # ── Step 3: Forensic Critic / QA Pass ───────────────────────────
    print("\n[STEP 3/3] Running Forensic Critic / QA Pass...")
    with open(obs_dir / "edit_decisions.json", "r", encoding="utf-8") as f:
        edit_decisions = json.load(f)

    qa = CriticQAEvaluator()
    qa_report = qa.evaluate(
        rendered_video_path=final_video,
        timeline_ir=timeline_ir,
        edit_decisions=edit_decisions,
        creative_plan=plan_dict,
        output_dir=obs_dir,
    )

    total_time = round(time.time() - t0, 2)

    # ── Summary Display ─────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PHASE 5 EXECUTION & QA RESULTS SUMMARY")
    print("=" * 75)
    print(f"Total Pipeline Runtime:      {total_time}s")
    print(f"Physical Render File:        {final_video}")
    print(f"Rendered Resolution:         {qa_report.get('resolution')} (9:16 Vertical)")
    print(f"Rendered Duration:           {qa_report.get('measured_duration'):.2f}s (Timeline: {timeline_ir.timeline_duration:.2f}s)")
    print(f"File Size:                   {qa_report.get('file_size_mb')} MB")
    print(f"Audio Mean Volume:           {qa_report.get('audio_mean_volume_db'):.1f} dB")
    print(f"Audio Peak Volume:           {qa_report.get('audio_max_volume_db'):.1f} dB (Clipping: {qa_report.get('clipping_detected')})")
    print(f"Operations Verified:         {qa_report['summary']['passed_operations']} / {qa_report['summary']['total_operations_verified']}")
    print(f"OVERALL QA VERDICT:          {qa_report.get('status')}")

    print("\nDECISION AUDIT BREAKDOWN:")
    for check in qa_report["decision_checks"]:
        v_str = "PASS" if check["passed"] else "FAIL"
        print(f"  [{v_str}] {check['operation']:<18} | {check.get('evidence')}")

    print("\nREPRESENTATIVE FRAME CAPTURES EXTRACTED:")
    for frame_p in qa_report.get("frame_captures", []):
        print(f"  - {Path(frame_p).name} ({round(Path(frame_p).stat().st_size / 1024, 1)} KB)")

    print("\nARTIFACTS GENERATED:")
    print(f"  1. Final Edited Reel:      {final_video}")
    print(f"  2. QA Validation Report:   {obs_dir / 'render_validation_report.json'}")
    print(f"  3. Interactive QA Report:  {obs_dir / 'render_qa_report.html'}")
    print(f"  4. Timeline CMX 3600 EDL:  {obs_dir / 'timeline.edl'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
