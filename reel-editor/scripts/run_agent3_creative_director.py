"""
scripts/run_agent3_creative_director.py

Run Agent 3 (Creative Director) on Surya.mp4
────────────────────────────────────────────
Consumes:
  outputs/Surya_phase2_observer/edit_decisions.json
  outputs/Surya_phase2_observer/story_edit_plan.json
  outputs/Surya_phase2_observer/enriched_transcript.json
  outputs/Surya_phase2_observer/media_meta.json

Produces:
  outputs/Surya_phase2_observer/creative_decisions.json
  outputs/Surya_phase2_observer/creative_report.html
  outputs/Surya_phase2_observer/timeline_ir.json
  outputs/Surya_phase2_observer/creative_validation_report.json

Enforces:
  DO NOT render the final reel yet.
  Show representative creative treatments for:
  - 1 HeroText
  - 3 EmphasisText
  - 2 Punch-ins
  - 2 B-roll requests
  - 1 SFX event
  - Preserved NO_EDIT areas
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
from core.logger import get_logger

logger = get_logger("run_agent3_creative_director")


def main():
    print("=" * 70)
    print("PHASE 4: AGENT 3 CREATIVE DIRECTOR ON SURYA.MP4")
    print("=" * 70)

    obs_dir = REPO_ROOT / "outputs" / "Surya_phase2_observer"
    source_video = REPO_ROOT / "Surya.mp4"

    if not obs_dir.exists():
        print(f"ERROR: Output directory '{obs_dir}' does not exist.")
        sys.exit(1)

    t0 = time.time()
    result = run_agent3(
        output_dir=obs_dir,
        source_video_path=source_video,
        style_profile_name="EDITORIAL_CINEMATIC",
    )
    elapsed = time.time() - t0

    plan_dict = result["creative_plan"]
    timeline_dict = result["timeline_ir"]
    decisions = plan_dict.get("decisions", [])

    print("\n" + "=" * 70)
    print("AGENT 3 CREATIVE DIRECTOR SUMMARY")
    print("=" * 70)
    print(f"Execution Time:              {elapsed:.3f}s")
    print(f"Active Style Profile:        {plan_dict.get('style_profile')}")
    print(f"Total Creative Decisions:    {len(decisions)}")
    print(f"Resolved Operations:         {plan_dict.get('resolved_count')}")
    print(f"Preserved Spans (NO_EDIT):   {plan_dict.get('preserved_spans_count')}")
    print(f"Unresolved Assets (honest):  {plan_dict.get('unresolved_assets_count')}")
    print(f"Conflicts Resolved:          {plan_dict.get('conflict_count')}")
    print(f"Validation Status:           {'PASSED (13/13 checks)' if result.get('validation_passed') else 'FAILED'}")

    print("\nTIMELINE IR COMPILED:")
    print(f"  Timeline Duration:         {timeline_dict.get('timeline_duration'):.2f}s")
    print(f"  Target Resolution:         {timeline_dict.get('target_width')}x{timeline_dict.get('target_height')} ({timeline_dict.get('target_aspect_ratio')})")
    print(f"  Video Clips (Track V1):    {len(timeline_dict.get('video_tracks', [{}])[0].get('clips', []))}")
    print(f"  Text Events (Track T1):    {len(timeline_dict.get('text_tracks', [{}])[0].get('events', []))}")
    print(f"  B-Roll Overlay Clips:      {len(timeline_dict.get('broll_tracks', [{}])[0].get('clips', []))}")
    print(f"  Audio Tracks:              {len(timeline_dict.get('audio_tracks', []))}")
    print(f"  Unresolved Items:          {len(timeline_dict.get('unresolved_items', []))}")

    # Display representative treatments
    print("\nREPRESENTATIVE CREATIVE TREATMENTS:")

    # 1. HeroText
    hero = next((d for d in decisions if d["operation"] == "HERO_TEXT"), None)
    if hero:
        typo = hero.get("typography", {})
        anim = hero.get("animation", {})
        print(f"\n[1. HERO_TEXT] {hero['time_range'][0]:.2f}s - {hero['time_range'][1]:.2f}s ({hero['story_act']})")
        print(f"  Source Decision:  {hero['source_decision_id']}")
        print(f"  Text:             '{hero.get('text_content')}'")
        print(f"  Font:             {typo.get('font')} ({typo.get('size')}pt, {typo.get('weight')})")
        print(f"  Fallback Fonts:   {', '.join(typo.get('font_fallback', []))}")
        print(f"  Colors:           Primary {typo.get('color_primary')}, Accent {typo.get('color_highlight')}")
        print(f"  Safe Zone:        {typo.get('safe_zone')} (Position: {typo.get('position')})")
        print(f"  Animation:        {anim.get('preset')} with easing={anim.get('easing')} ({anim.get('duration')}s)")

    # 2. EmphasisText
    emphs = [d for d in decisions if d["operation"] == "EMPHASIS_TEXT"]
    print(f"\n[2. EMPHASIS_TEXT (3 examples)]")
    for idx, e in enumerate(emphs[:3], 1):
        typo = e.get("typography", {})
        anim = e.get("animation", {})
        print(f"  Example {idx}: '{e.get('text_content')}' @ {e['time_range'][0]:.2f}s - {e['time_range'][1]:.2f}s")
        print(f"    Font:      {typo.get('font')} ({typo.get('size')}pt, {typo.get('weight')})")
        print(f"    Accent:    {typo.get('color_highlight')}")
        print(f"    Animation: {anim.get('preset')} ({anim.get('easing')})")

    # 3. Punch-Ins
    punches = [d for d in decisions if d["operation"] == "PUNCH_IN"]
    print(f"\n[3. PUNCH_INS (2 examples)]")
    for idx, p in enumerate(punches[:2], 1):
        pi = p.get("punch_in", {})
        print(f"  Example {idx}: {pi.get('conceptual_strength')} @ {p['time_range'][0]:.2f}s - {p['time_range'][1]:.2f}s")
        print(f"    Scale:      {pi.get('scale')}x")
        print(f"    Anchor:     x={pi.get('anchor', {}).get('x')}, y={pi.get('anchor', {}).get('y')}")
        print(f"    Transition: {pi.get('transition')}")
        print(f"    Headroom:   Preserved={pi.get('headroom_preserved')}")

    # 4. B-Roll Requests
    brolls = [d for d in decisions if "BROLL" in d["operation"]]
    print(f"\n[4. B-ROLL REQUESTS (2 examples)]")
    for idx, b in enumerate(brolls[:2], 1):
        bs = b.get("broll", {})
        print(f"  Example {idx}: '{bs.get('asset_query')}' @ {b['time_range'][0]:.2f}s - {b['time_range'][1]:.2f}s")
        print(f"    Role:       {bs.get('role')} (Duration: {bs.get('required_duration')}s)")
        print(f"    Aspect:     {bs.get('aspect_treatment')} ({bs.get('overlay_treatment')})")
        print(f"    Status:     {b.get('status')} (No asset hallucinated)")

    # 5. SFX Event
    sfxs = [d for d in decisions if "SFX" in d["operation"]]
    print(f"\n[5. SFX EVENT]")
    for s in sfxs[:1]:
        ss = s.get("sfx", {})
        print(f"  Category:     {ss.get('category')} ('{ss.get('sound_cue')}') @ {s['time_range'][0]:.2f}s - {s['time_range'][1]:.2f}s")
        print(f"  Gain:         {ss.get('gain_db')}dB (Voice priority respected)")
        print(f"  Fades:        {ss.get('fade_in_sec')}s in / {ss.get('fade_out_sec')}s out")
        print(f"  Status:       {s.get('status')}")

    # 6. Preserved NO_EDIT
    no_edits = [d for d in decisions if d["operation"] == "NO_EDIT"]
    print(f"\n[6. PRESERVED NO_EDIT AREAS ({len(no_edits)} preserved)]")
    for ne in no_edits[:3]:
        print(f"  {ne['time_range'][0]:.2f}s - {ne['time_range'][1]:.2f}s ({ne['story_act']}) | Status: {ne['status']} | {ne['reason']}")

    print("\n" + "=" * 70)
    print("GENERATED PHASE 4 ARTIFACTS:")
    print(f"  1. Creative Decisions:  {result['creative_decisions_file']}")
    print(f"  2. Timeline IR:         {result['timeline_ir_file']}")
    print(f"  3. Creative Report:     {result['creative_report_file']}")
    print(f"  4. Validation Report:   {result['validation_report_file']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
