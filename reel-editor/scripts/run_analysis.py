"""
scripts/run_analysis.py

Main CLI entry point for Agent 1 — Raw Reel Analyzer.

Usage:
  python scripts/run_analysis.py --video path/to/reel.mp4
  python scripts/run_analysis.py --video reel.mp4 --instruction "Make this serious and cinematic"
  python scripts/run_analysis.py --video reel.mp4 --language hinglish --preset balanced
  python scripts/run_analysis.py --video reel.mp4 --stages 1,2   (run only specific stages)

Output (in outputs/<video_stem>_<timestamp>/):
  media_meta.json        — Stage 1
  transcript_raw.json    — Stage 2
  audio_features.json    — Stage 3  (coming)
  visual_features.json   — Stage 4  (coming)
  fused_timeline.json    — Stage 5  (coming)
  story_analysis.json    — Stage 6  (coming)
  opportunity_inventory.json — Stage 7  (coming)
  edit_blueprint.json    — Stage 8  (coming)
  analysis_report.html   — Stage 8  (coming)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.logger import setup_logger, get_logger
from core.hardware_detector import detect_hardware
from core.model_selector import select_models

setup_logger(verbose=False)
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Reel Editor — Agent 1 Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", required=True, help="Path to the video file to analyze")
    parser.add_argument(
        "--instruction", default=None,
        help='Optional creative instruction, e.g. "Make this serious and cinematic"'
    )
    parser.add_argument(
        "--language", default="auto",
        choices=["auto", "en", "hi", "hinglish"],
        help="Language hint for transcription (default: auto-detect)"
    )
    parser.add_argument(
        "--preset", default="balanced",
        choices=["fast", "balanced", "thorough"],
        help="Quality preset (default: balanced)"
    )
    parser.add_argument(
        "--stages", default=None,
        help="Comma-separated list of stage numbers to run, e.g. '1,2' (default: all)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Custom output directory (default: outputs/<video_stem>_<timestamp>/)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show debug output"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(verbose=args.verbose)

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "outputs" / f"{video_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Determine which stages to run
    if args.stages:
        stages_to_run = {int(s.strip()) for s in args.stages.split(",")}
    else:
        stages_to_run = {1, 2, 3, 4, 5, 6, 7, 8}

    # Load hardware profile (from last diagnose.py run, or detect fresh)
    hw_profile_path = _PROJECT_ROOT / "outputs" / "hardware_profile.json"
    if hw_profile_path.exists():
        with open(hw_profile_path) as f:
            hw_data = json.load(f)
        hw = hw_data["hardware"]
        deps = hw_data["dependencies"]
        models = hw_data["models"]
        logger.info(f"Using hardware profile: {models['profile_key']}")
    else:
        logger.info("No hardware_profile.json found. Running detection now...")
        hw = detect_hardware()
        deps = {"ffmpeg_available": True, "ollama_running": True, "ollama_models": []}
        models = select_models(hw, deps)

    run_config = {
        "video_path": str(video_path),
        "user_instruction": args.instruction,
        "language_hint": args.language,
        "quality_preset": args.preset,
        "whisper_model": models["whisper_model"],
        "whisper_device": models["whisper_device"],
        "whisper_compute_type": models["whisper_compute_type"],
        "llm_model": models["llm_model"],
        "llm_available": models["llm_available"],
        "frame_sample_fps": models["frame_sample_fps"],
        "output_dir": str(output_dir),
        "stages_to_run": sorted(stages_to_run),
        "started_at": datetime.now().isoformat(),
    }

    # Save run config
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"  AGENT 1 — RAW REEL ANALYZER")
    logger.info(f"  Video:    {video_path.name}")
    logger.info(f"  Whisper:  {models['whisper_model']} ({models['whisper_device']})")
    logger.info(f"  LLM:      {models['llm_model']} (available={models['llm_available']})")
    logger.info(f"  Language: {args.language}")
    logger.info(f"  Stages:   {sorted(stages_to_run)}")
    logger.info("=" * 60)

    stage_timings = {}
    media_meta = None
    transcript = None

    # ── Stage 1 ────────────────────────────────────────────────────
    if 1 in stages_to_run:
        t0 = time.time()
        logger.info("\n--- STAGE 1: MEDIA EXTRACTION ---")
        from agents.agent1_analyzer.stages.s1_media_extraction import run as run_s1
        media_meta = run_s1(
            video_path=video_path,
            output_dir=output_dir,
            frame_sample_fps=models["frame_sample_fps"],
        )
        stage_timings["s1_media_extraction"] = round(time.time() - t0, 2)
        logger.info(f"Stage 1 done in {stage_timings['s1_media_extraction']:.1f}s")
    else:
        # Load from previous run if skipping
        meta_path = output_dir / "media_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                media_meta = json.load(f)

    # ── Stage 2 ────────────────────────────────────────────────────
    if 2 in stages_to_run and media_meta:
        if not media_meta.get("has_audio", True):
            logger.warning("No audio track detected. Skipping Stage 2.")
        else:
            t0 = time.time()
            logger.info("\n--- STAGE 2: SPEECH EXTRACTION ---")
            audio_path = Path(media_meta["audio_path"])
            from agents.agent1_analyzer.stages.s2_speech_extraction import run as run_s2
            transcript = run_s2(
                audio_path=audio_path,
                output_dir=output_dir,
                whisper_model=models["whisper_model"],
                whisper_device=models["whisper_device"],
                whisper_compute_type=models["whisper_compute_type"],
                language_hint=args.language if args.language != "auto" else None,
            )
            stage_timings["s2_speech_extraction"] = round(time.time() - t0, 2)
            logger.info(f"Stage 2 done in {stage_timings['s2_speech_extraction']:.1f}s")
    else:
        tx_path = output_dir / "transcript_raw.json"
        if tx_path.exists():
            with open(tx_path) as f:
                transcript = json.load(f)

    # ── Stage 3 ────────────────────────────────────────────────────
    audio_features = None
    if 3 in stages_to_run and transcript and media_meta:
        t0 = time.time()
        logger.info("\n--- STAGE 3: AUDIO SIGNAL ANALYSIS ---")
        audio_path = Path(media_meta["audio_path"])
        from agents.agent1_analyzer.stages.s3_audio_features import run as run_s3
        audio_features = run_s3(
            audio_path=audio_path,
            transcript=transcript,
            output_dir=output_dir,
        )
        stage_timings["s3_audio_features"] = round(time.time() - t0, 2)
        logger.info(f"Stage 3 done in {stage_timings['s3_audio_features']:.1f}s")
    else:
        af_path = output_dir / "audio_features.json"
        if af_path.exists():
            with open(af_path) as f:
                audio_features = json.load(f)

    # ── Stage 4 ────────────────────────────────────────────────────
    visual_features = None
    if 4 in stages_to_run and media_meta:
        t0 = time.time()
        logger.info("\n--- STAGE 4: VISUAL CV ANALYSIS ---")
        frames_dir = Path(media_meta["frames_dir"])
        from agents.agent1_analyzer.stages.s4_visual_cv import run as run_s4
        visual_features = run_s4(
            frames_dir=frames_dir,
            media_meta=media_meta,
            output_dir=output_dir,
            mediapipe_complexity=models.get("mediapipe_complexity", 0),
        )
        stage_timings["s4_visual_cv"] = round(time.time() - t0, 2)
        logger.info(f"Stage 4 done in {stage_timings['s4_visual_cv']:.1f}s")
    else:
        vf_path = output_dir / "visual_features.json"
        if vf_path.exists():
            with open(vf_path) as f:
                visual_features = json.load(f)

    # ── Stage 5 ────────────────────────────────────────────────────
    fused_timeline = None
    if 5 in stages_to_run:
        t0 = time.time()
        logger.info("\n--- STAGE 5: FEATURE FUSION & TIMELINE SEGMENTATION ---")
        from agents.agent1_analyzer.stages.s5_feature_fusion import run as run_s5
        fused_timeline = run_s5(
            output_dir=output_dir,
            transcript=transcript,
            audio_features=audio_features,
            visual_features=visual_features,
            media_meta=media_meta,
        )
        stage_timings["s5_feature_fusion"] = round(time.time() - t0, 2)
        logger.info(f"Stage 5 done in {stage_timings['s5_feature_fusion']:.1f}s")
    else:
        ft_path = output_dir / "fused_timeline.json"
        if ft_path.exists():
            with open(ft_path) as f:
                fused_timeline = json.load(f)

    # Stages 6–8 not yet implemented
    unimplemented = stages_to_run - {1, 2, 3, 4, 5}
    if unimplemented:
        logger.info(f"\nStages {sorted(unimplemented)} not yet implemented (coming in Days 4–10).")

    # ── Print summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ANALYSIS SUMMARY")
    logger.info("=" * 60)

    if media_meta:
        logger.info(f"  Duration:    {media_meta['duration_seconds']:.1f}s")
        logger.info(f"  Resolution:  {media_meta['width']}x{media_meta['height']} ({media_meta['aspect_ratio']})")
        logger.info(f"  Scenes:      {media_meta['scene_info'].get('scene_count', 'n/a')}")

    if transcript:
        stats = transcript["speech_stats"]
        logger.info(f"  Words:       {stats['total_words']}")
        logger.info(f"  Sentences:   {stats['total_sentences']}")
        logger.info(f"  Avg WPM:     {stats['average_wpm']}")
        logger.info(f"  Fillers:     {stats['filler_word_count']} ({stats['filler_percentage']}%)")
        logger.info(f"  Language:    {stats['detected_language']} (p={stats['language_probability']})")
        logger.info(f"  Silence:     {stats['silence_duration_seconds']:.1f}s")
        logger.info(f"\n  TRANSCRIPT PREVIEW:")
        preview = transcript["full_text"][:300]
        logger.info(f"  \"{preview}{'...' if len(transcript['full_text']) > 300 else ''}\"")

    logger.info(f"\n  Output dir:  {output_dir}")
    logger.info("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
