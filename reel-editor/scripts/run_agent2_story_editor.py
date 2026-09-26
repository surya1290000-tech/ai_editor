"""
scripts/run_agent2_story_editor.py

Execute Agent 2: Story Editor on Surya.mp4 (Phase 3)
────────────────────────────────────────────────────
Consumes:
  - outputs/Surya_phase2_observer/evidence_inventory.json
  - outputs/Surya_phase2_observer/story_observations.json
  - outputs/Surya_phase2_observer/opportunity_inventory.json
  - outputs/Surya_phase2_observer/enriched_transcript.json
  - outputs/Surya_phase2_observer/media_meta.json

Produces:
  - outputs/Surya_phase2_observer/story_edit_plan.json
  - outputs/Surya_phase2_observer/edit_decisions.json
  - outputs/Surya_phase2_observer/editorial_report.html
  - outputs/Surya_phase2_observer/editorial_validation_report.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger
from agents.agent2_director.creative_director import run as run_agent2

logger = get_logger(__name__)


def main():
    print("=" * 70)
    print("PHASE 3: AGENT 2 STORY EDITOR RUN ON SURYA.MP4")
    print("=" * 70)

    t0 = time.time()
    work_dir = PROJECT_ROOT / "outputs" / "Surya_phase2_observer"
    budget_cfg = PROJECT_ROOT / "config" / "edit_budget.yaml"

    if not (work_dir / "opportunity_inventory.json").exists():
        print(f"Error: Phase 2 artifacts not found in {work_dir}")
        return 1

    decisions_dict = run_agent2(
        output_dir=work_dir,
        pacing_preset="balanced",
        budget_config_path=budget_cfg,
    )

    total_time = round(time.time() - t0, 3)

    # Load outputs for console summary
    with open(work_dir / "story_edit_plan.json", "r", encoding="utf-8") as f:
        story_plan = json.load(f)
    with open(work_dir / "edit_decisions.json", "r", encoding="utf-8") as f:
        edit_doc = json.load(f)
    with open(work_dir / "editorial_validation_report.json", "r", encoding="utf-8") as f:
        val_report = json.load(f)

    summary = story_plan.get("decisions_summary", {})
    b_util = edit_doc.get("budget_utilization", {})

    print("\n" + "=" * 70)
    print("AGENT 2 EDITORIAL DECISION SUMMARY")
    print("=" * 70)
    print(f"Execution Time:              {total_time}s")
    print(f"Total Opportunities Evaluated: {summary.get('total_opportunities_evaluated', 0)}")
    print(f"Accepted Edit Interventions:  {summary.get('total_accepted_interventions', 0)}")
    print(f"NO_EDIT Preserved Spans:      {summary.get('total_no_edit_preserved', 0)}")
    print(f"Rejected Opportunities:       {summary.get('total_rejected', 0)}")
    print(f"Selectivity Ratio:            {summary.get('selectivity_ratio', 'N/A')}")
    print(f"Validation Status:            {'PASSED (10/10 checks)' if val_report.get('passed') else 'FAILED'}")

    print("\nBUDGET UTILIZATION:")
    print(f"  Hero Text:     {b_util.get('hero_text', {}).get('used', 0)} / {b_util.get('hero_text', {}).get('limit', 2)}")
    punch_info = b_util.get('punch_ins', {})
    print(f"  Punch-Ins:     {punch_info.get('total_used', 0)} (Strong: {punch_info.get('strong', {}).get('used', 0)}, Medium: {punch_info.get('medium', {}).get('used', 0)}, Subtle: {punch_info.get('subtle', {}).get('used', 0)})")
    print(f"  B-Roll Clips:  {b_util.get('broll', {}).get('used', 0)} / {b_util.get('broll', {}).get('limit', 3)}")
    print(f"  SFX Cues:      {b_util.get('sfx', {}).get('used', 0)} / {b_util.get('sfx', {}).get('limit', 6)}")
    print(f"  Emphasis Text: {b_util.get('emphasis_text', {}).get('used', 0)} / {b_util.get('emphasis_text', {}).get('limit', 8)}")

    print("\nACCEPTED EDIT INTERVENTIONS:")
    for d in edit_doc.get("accepted_decisions", []):
        param = f" [strength: {d.get('strength')}]" if d.get('strength') else f" [text: {d.get('text_content')}]" if d.get('text_content') else f" [concept: {d.get('concept_query')[:25]}...]" if d.get('concept_query') else f" [cue: {d.get('sound_cue')}]" if d.get('sound_cue') else ""
        print(f"  [{d['operation']:<18}] {d['start']:>5.2f}s – {d['end']:>5.2f}s ({d['story_act']:<11}) | {d['reason'][:55]}...{param}")

    print("\nSAMPLE REJECTED OPPORTUNITIES:")
    for d in edit_doc.get("rejected_decisions", [])[:6]:
        print(f"  [{d['operation']:<18}] {d['start']:>5.2f}s – {d['end']:>5.2f}s | {d['reason']}")

    print("\nSAMPLE NO_EDIT / PRESERVED SPANS:")
    for d in edit_doc.get("no_edit_decisions", [])[:6]:
        print(f"  [{d['operation']:<18}] {d['start']:>5.2f}s – {d['end']:>5.2f}s | {d['reason']}")

    print("\nGENERATED ARTIFACTS:")
    print(f"  1. Story Edit Plan:    {work_dir / 'story_edit_plan.json'}")
    print(f"  2. Edit Decisions:     {work_dir / 'edit_decisions.json'}")
    print(f"  3. Editorial Report:   {work_dir / 'editorial_report.html'}")
    print(f"  4. Validation Report:  {work_dir / 'editorial_validation_report.json'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
