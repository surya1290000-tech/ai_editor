"""
scripts/run_agent1_observer.py

Execute Agent 1 Observer Pipeline on Surya.mp4 (Phase 2)
────────────────────────────────────────────────────────
Produces:
  1. enriched_transcript.json
  2. evidence_inventory.json
  3. story_observations.json
  4. opportunity_inventory.json
  5. analysis_report.html
  6. evidence_validation_report.json
"""

import json
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import get_logger
from core.json_utils import save_json, load_json
from core.evidence.pipeline import run_agent1_observer_pipeline
from agents.agent1_analyzer.stages.s3_audio_features import run as run_s3
from agents.agent1_analyzer.stages.s5_feature_fusion import run as run_s5

logger = get_logger(__name__)


def main():
    print("=" * 70)
    print("PHASE 2: AGENT 1 OBSERVER PIPELINE RUN ON SURYA.MP4")
    print("=" * 70)

    t0 = time.time()
    source_dir = PROJECT_ROOT / "outputs" / "Surya_20260912_214941"
    output_dir = PROJECT_ROOT / "outputs" / "Surya_phase2_observer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Base Assets
    media_meta_path = source_dir / "media_meta.json"
    with open(media_meta_path, "r", encoding="utf-8") as f:
        media_meta = json.load(f)

    # Load fresh Phase 1 multilingual transcript
    phase1_tx_path = PROJECT_ROOT / "outputs" / "benchmark_asr" / "transcript_raw.json"
    if phase1_tx_path.exists():
        with open(phase1_tx_path, "r", encoding="utf-8") as f:
            transcript_raw = json.load(f)
    else:
        with open(source_dir / "transcript_raw.json", "r", encoding="utf-8") as f:
            transcript_raw = json.load(f)

    # Copy raw transcript into output_dir
    save_json(transcript_raw, output_dir / "transcript_raw.json")
    save_json(media_meta, output_dir / "media_meta.json")

    # Copy audio.wav & visual_features
    audio_wav_path = source_dir / "audio.wav"
    vf_path = source_dir / "visual_features.json"
    with open(vf_path, "r", encoding="utf-8") as f:
        visual_features = json.load(f)
    save_json(visual_features, output_dir / "visual_features.json")

    # 2. Compute audio features on the multilingual transcript
    print("\n[1/4] Aligning audio features with multilingual transcript...")
    audio_features = run_s3(
        audio_path=audio_wav_path,
        transcript=transcript_raw,
        output_dir=output_dir,
    )

    # 3. Fuse features
    print("[2/4] Fusing multi-modal timeline...")
    fused_timeline = run_s5(
        output_dir=output_dir,
        transcript=transcript_raw,
        audio_features=audio_features,
        visual_features=visual_features,
        media_meta=media_meta,
    )

    # 4. Run Stage 6 Story Analysis or heuristic
    story_analysis_path = source_dir / "story_analysis.json"
    story_analysis = None
    if story_analysis_path.exists():
        with open(story_analysis_path, "r", encoding="utf-8") as f:
            story_analysis = json.load(f)

    # 5. Execute Agent 1 Observer Evidence Pipeline
    print("[3/4] Synthesizing evidence inventory, story acts, & opportunities...")
    results = run_agent1_observer_pipeline(
        output_dir=output_dir,
        media_meta=media_meta,
        transcript_raw=transcript_raw,
        audio_features=audio_features,
        visual_features=visual_features,
        fused_timeline=fused_timeline,
        story_analysis=story_analysis,
    )

    total_time = round(time.time() - t0, 2)
    ev_inv = results["evidence_inventory"]
    story_obs = results["story_observations"]
    opp_inv = results["opportunity_inventory"]
    val_report = results["validation_results"]
    enriched_tx = results["enriched_transcript"]

    # 6. Display Summary Report
    print("\n" + "=" * 70)
    print("PHASE 2 OBSERVER EXECUTION COMPLETE")
    print("=" * 70)
    print(f"Total Runtime:       {total_time}s")
    print(f"Enriched Words:      {len(enriched_tx.words)}")
    print(f"Languages:           {enriched_tx.detected_languages}")
    print(f"Evidence Items:      {len(ev_inv.evidence)}")
    print(f"Story Acts:          {len(story_obs.story_observations)}")
    print(f"Edit Opportunities:  {len(opp_inv.opportunities)}")
    print(f"Validation Status:   {'PASSED' if val_report['passed'] else 'FAILED'}")

    print("\nEVIDENCE TYPES BREAKDOWN:")
    for t, c in sorted(ev_inv.metadata.get("types_count", {}).items()):
        print(f"  - {t:<28}: {c}")

    print("\nSTORY ACTS:")
    for so in story_obs.story_observations:
        print(f"  [{so.type:<12}] {so.time_range[0]:.1f}s – {so.time_range[1]:.1f}s (conf: {so.confidence:.2f}) | {so.summary[:60]}...")

    print("\nOPPORTUNITY TYPES BREAKDOWN:")
    for ot, oc in sorted(opp_inv.metadata.get("types_count", {}).items()):
        print(f"  - {ot:<28}: {oc}")

    print("\nSAVED ARTIFACTS:")
    print(f"  1. Enriched Transcript:   {output_dir / 'enriched_transcript.json'}")
    print(f"  2. Evidence Inventory:    {output_dir / 'evidence_inventory.json'}")
    print(f"  3. Story Observations:    {output_dir / 'story_observations.json'}")
    print(f"  4. Opportunity Inventory: {output_dir / 'opportunity_inventory.json'}")
    print(f"  5. HTML Timeline Report:  {output_dir / 'analysis_report.html'}")
    print(f"  6. Validation Report:     {output_dir / 'evidence_validation_report.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
