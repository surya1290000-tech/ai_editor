"""
scripts/run_generalization_test.py

Executes the autonomous ProductionPipeline on the new Telugu video:
inputs/whatsapp_telugu_reel.mp4
Outputs to:
outputs/whatsapp_telugu_reel_generalization
"""

import sys
import io
import json
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import ProductionPipeline

def main():
    video_path = PROJECT_ROOT / "inputs" / "whatsapp_telugu_reel.mp4"
    output_dir = PROJECT_ROOT / "outputs" / "whatsapp_telugu_reel_generalization"

    if not video_path.exists():
        print(f"Error: Target video not found at {video_path}")
        sys.exit(1)

    print("==================================================")
    print("STARTING AUTONOMOUS GENERALIZATION RUN")
    print(f"Video:  {video_path}")
    print(f"Output: {output_dir}")
    print("==================================================")

    pipeline = ProductionPipeline(workspace_root=PROJECT_ROOT)
    summary = pipeline.run(
        video_path=video_path,
        output_dir=output_dir,
        style_preset="EDITORIAL_CINEMATIC",
        pacing_preset="balanced",
    )

    summary_file = output_dir / "generalization_run_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================")
    print("GENERALIZATION RUN COMPLETE!")
    print(f"Summary written to: {summary_file}")
    print(json.dumps(summary, indent=2))
    print("==================================================")

if __name__ == "__main__":
    main()
