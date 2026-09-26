"""
core/workflow/node_executors.py

Wraps existing Stage 1–5 analysis modules as WorkflowNode executors.
Each function has the signature: (ctx: ExecutionContext) -> dict[str, Any]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.workflow.engine import ExecutionContext


def execute_s1_media(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 1: Media Extraction — FFmpeg + PySceneDetect."""
    from agents.agent1_analyzer.stages.s1_media_extraction import run as run_s1

    video_path = Path(ctx.inputs.get("video_file", ctx.video_path))
    ctx.log("Starting media extraction...")
    ctx.set_progress(10, "Probing metadata")

    # Load config for frame sample fps
    frame_fps = ctx.config.get("frame_sample_fps", 1)

    media_meta = run_s1(
        video_path=video_path,
        output_dir=ctx.output_dir,
        frame_sample_fps=frame_fps,
    )

    ctx.set_progress(90, "Saving metadata")
    ctx.set_metrics({
        "duration": f"{media_meta['duration_seconds']:.1f}s",
        "resolution": f"{media_meta['width']}x{media_meta['height']}",
        "fps": media_meta["fps"],
        "aspect_ratio": media_meta["aspect_ratio"],
        "scenes": media_meta["scene_info"]["scene_count"],
    })
    ctx.add_artifact("media_meta.json", str(ctx.output_dir / "media_meta.json"), "json")
    ctx.add_artifact("audio.wav", str(ctx.output_dir / "audio.wav"), "audio")

    node = ctx._event_bus  # Update node execution metadata
    return {
        "audio_wav": str(ctx.output_dir / "audio.wav"),
        "frames_dir": str(ctx.output_dir / "frames"),
        "media_meta": media_meta,
    }


def execute_s2_speech(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 2: Speech Extraction — faster-whisper."""
    from agents.agent1_analyzer.stages.s2_speech_extraction import run as run_s2

    audio_path = Path(ctx.inputs["audio_wav"])
    ctx.log("Loading Whisper model...")
    ctx.set_progress(5, "Loading model")

    whisper_model = ctx.config.get("whisper_model", "base")
    whisper_device = ctx.config.get("whisper_device", "cpu")
    whisper_compute = ctx.config.get("whisper_compute_type", "int8")
    language_hint = ctx.config.get("language_hint")

    transcript = run_s2(
        audio_path=audio_path,
        output_dir=ctx.output_dir,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute,
        language_hint=language_hint,
    )

    stats = transcript.get("speech_stats", {})
    ctx.set_metrics({
        "words": stats.get("total_words", 0),
        "sentences": stats.get("total_sentences", 0),
        "wpm": stats.get("average_wpm", 0),
        "language": stats.get("detected_language", "?"),
        "fillers": stats.get("filler_word_count", 0),
    })
    ctx.add_artifact("transcript_raw.json", str(ctx.output_dir / "transcript_raw.json"), "json")
    ctx.set_progress(100, "Transcription complete")

    return {"transcript_raw": transcript}


def execute_s3_audio(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 3: Audio Signal Analysis — librosa."""
    from agents.agent1_analyzer.stages.s3_audio_features import run as run_s3

    audio_path = Path(ctx.inputs["audio_wav"])
    transcript = ctx.inputs.get("transcript_raw")

    ctx.log("Computing audio features...")
    ctx.set_progress(10, "Loading audio")

    # If transcript is a path string, load it
    if isinstance(transcript, str):
        with open(transcript, "r", encoding="utf-8") as f:
            transcript = json.load(f)

    audio_features = run_s3(
        audio_path=audio_path,
        transcript=transcript,
        output_dir=ctx.output_dir,
    )

    ctx.set_metrics({
        "baseline_energy": audio_features.get("speaker_baseline_energy", 0),
        "baseline_pitch_hz": audio_features.get("speaker_baseline_pitch_hz", 0),
        "baseline_wpm": audio_features.get("speaker_baseline_wpm", 0),
    })
    ctx.add_artifact("audio_features.json", str(ctx.output_dir / "audio_features.json"), "json")
    ctx.set_progress(100, "Audio analysis complete")

    return {"audio_features": audio_features}


def execute_s4_visual(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 4: Visual CV Analysis — MediaPipe + OpenCV."""
    from agents.agent1_analyzer.stages.s4_visual_cv import run as run_s4

    frames_dir = Path(ctx.inputs["frames_dir"])
    media_meta = ctx.inputs["media_meta"]

    ctx.log("Analyzing frames with MediaPipe + OpenCV...")
    ctx.set_progress(10, "Initializing face detector")

    # If media_meta is a path, load it
    if isinstance(media_meta, str):
        with open(media_meta, "r", encoding="utf-8") as f:
            media_meta = json.load(f)

    complexity = ctx.config.get("mediapipe_complexity", 0)

    visual_features = run_s4(
        frames_dir=frames_dir,
        media_meta=media_meta,
        output_dir=ctx.output_dir,
        mediapipe_complexity=complexity,
    )

    ctx.set_metrics({
        "frames_analyzed": visual_features.get("frames_analyzed", 0),
        "dominant_framing": visual_features.get("dominant_framing", "?"),
        "face_detected": f"{visual_features.get('face_detected_ratio', 0):.0%}",
        "shot_segments": len(visual_features.get("shot_segments", [])),
    })
    ctx.add_artifact("visual_features.json", str(ctx.output_dir / "visual_features.json"), "json")
    ctx.set_progress(100, "Visual analysis complete")

    return {"visual_features": visual_features}


def execute_s5_fusion(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 5: Feature Fusion — Pure Python deterministic merge."""
    from agents.agent1_analyzer.stages.s5_feature_fusion import run as run_s5

    ctx.log("Fusing multi-modal signals...")
    ctx.set_progress(10, "Aligning timelines")

    # Inputs may be dicts (from upstream) or need loading from files
    transcript = ctx.inputs.get("transcript_raw")
    audio_features = ctx.inputs.get("audio_features")
    visual_features = ctx.inputs.get("visual_features")
    media_meta = ctx.inputs.get("media_meta")

    fused_timeline = run_s5(
        output_dir=ctx.output_dir,
        transcript=transcript,
        audio_features=audio_features,
        visual_features=visual_features,
        media_meta=media_meta,
    )

    sentence_count = len(fused_timeline.get("sentences", []))
    pauses = fused_timeline.get("silence_regions", [])
    dead_spaces = sum(1 for p in pauses if p.get("category") == "DEAD_SPACE")

    ctx.set_metrics({
        "sentences_fused": sentence_count,
        "pauses_classified": len(pauses),
        "dead_spaces": dead_spaces,
    })
    ctx.add_artifact("fused_timeline.json", str(ctx.output_dir / "fused_timeline.json"), "json")
    ctx.add_artifact("llm_input.json", str(ctx.output_dir / "llm_input.json"), "json")
    ctx.set_progress(100, "Feature fusion complete")

    # Load llm_input from file since run_s5 saves it
    llm_input_path = ctx.output_dir / "llm_input.json"
    llm_input = None
    if llm_input_path.exists():
        with open(llm_input_path, "r", encoding="utf-8") as f:
            llm_input = json.load(f)

    return {
        "fused_timeline": fused_timeline,
        "llm_input": llm_input,
    }



def execute_s6_story(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 6: Story & Semantic Arc — Local Ollama LLM."""
    from agents.agent1_analyzer.stages.s6_story_analysis import run as run_s6

    ctx.log("Analyzing story arc with local LLM...")
    ctx.set_progress(10, "Initializing prompt")

    llm_input = ctx.inputs.get("llm_input")
    fused_timeline = ctx.inputs.get("fused_timeline")

    model_name = ctx.config.get("ollama_model", "llama3.2:latest")

    story_analysis = run_s6(
        output_dir=ctx.output_dir,
        llm_input=llm_input,
        fused_timeline=fused_timeline,
        model_name=model_name,
    )

    hook = story_analysis.get("hook_analysis", {})
    acts = story_analysis.get("narrative_acts", [])
    peaks = story_analysis.get("semantic_peaks", [])

    ctx.set_metrics({
        "hook_score": f"{hook.get('hook_strength_score', 0)}/10",
        "acts": len(acts),
        "semantic_peaks": len(peaks),
        "dropoff_risks": len(story_analysis.get("dropoff_risk_moments", [])),
    })
    ctx.add_artifact("story_analysis.json", str(ctx.output_dir / "story_analysis.json"), "json")
    ctx.set_progress(100, "Story analysis complete")

    return {"story_analysis": story_analysis}


def execute_s7_opportunities(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 7: Evidence Synthesis & Opportunity Engine (Phase 2 Standardized)."""
    from agents.agent1_analyzer.stages.s7_opportunity_engine import run as run_s7

    ctx.log("Generating multi-modal evidence inventory and edit opportunities...")
    ctx.set_progress(10, "Evaluating multi-modal evidence")

    fused_timeline = ctx.inputs.get("fused_timeline")
    story_analysis = ctx.inputs.get("story_analysis")
    media_meta = ctx.inputs.get("media_meta")
    transcript_raw = ctx.inputs.get("transcript_raw")
    audio_features = ctx.inputs.get("audio_features")
    visual_features = ctx.inputs.get("visual_features")

    opportunity_inventory = run_s7(
        output_dir=ctx.output_dir,
        fused_timeline=fused_timeline,
        story_analysis=story_analysis,
        media_meta=media_meta,
        transcript_raw=transcript_raw,
        audio_features=audio_features,
        visual_features=visual_features,
    )

    metadata = opportunity_inventory.get("metadata", {})
    types_count = metadata.get("types_count", {})

    ctx.set_metrics({
        "total_opps": opportunity_inventory.get("total_opportunities", 0),
        "cuts": types_count.get("cut_opportunity", 0),
        "punches": types_count.get("punch_in_opportunity", 0),
        "sfx": types_count.get("sfx_opportunity", 0),
        "b_roll": types_count.get("broll_opportunity", 0),
        "no_edits": types_count.get("no_edit_opportunity", 0),
    })
    ctx.add_artifact("enriched_transcript.json", str(ctx.output_dir / "enriched_transcript.json"), "json")
    ctx.add_artifact("evidence_inventory.json", str(ctx.output_dir / "evidence_inventory.json"), "json")
    ctx.add_artifact("story_observations.json", str(ctx.output_dir / "story_observations.json"), "json")
    ctx.add_artifact("opportunity_inventory.json", str(ctx.output_dir / "opportunity_inventory.json"), "json")
    ctx.add_artifact("analysis_report.html", str(ctx.output_dir / "analysis_report.html"), "html")
    ctx.set_progress(100, "Evidence synthesis and opportunity generation complete")

    return {"opportunity_inventory": opportunity_inventory}


def execute_s8_blueprint(ctx: ExecutionContext) -> dict[str, Any]:
    """Stage 8: Edit Blueprint & HTML Report Generator."""
    from agents.agent1_analyzer.stages.s8_blueprint_generator import run as run_s8

    ctx.log("Assembling final Edit Blueprint & interactive HTML report...")
    ctx.set_progress(10, "Building blueprint")

    opportunity_inventory = ctx.inputs.get("opportunity_inventory")
    story_analysis = ctx.inputs.get("story_analysis")
    fused_timeline = ctx.inputs.get("fused_timeline")

    result = run_s8(
        output_dir=ctx.output_dir,
        opportunity_inventory=opportunity_inventory,
        story_analysis=story_analysis,
        fused_timeline=fused_timeline,
    )

    blueprint = result.get("edit_blueprint", {})
    metrics = blueprint.get("metrics_summary", {})

    ctx.set_metrics({
        "final_duration": f"{metrics.get('estimated_trimmed_duration_seconds', 0):.1f}s",
        "total_saved": f"{metrics.get('total_seconds_saved', 0):.2f}s",
        "actions_ready": metrics.get("total_opportunities", 0),
    })
    ctx.add_artifact("edit_blueprint.json", str(ctx.output_dir / "edit_blueprint.json"), "json")
    ctx.add_artifact("report.html", str(ctx.output_dir / "report.html"), "html")
    ctx.set_progress(100, "Blueprint generation complete")

    return {
        "edit_blueprint": blueprint,
        "report_html": result.get("report_html_path"),
    }



def execute_agent2_director(ctx: ExecutionContext) -> dict[str, Any]:
    """Agent 2: Creative Director — Formulate concrete edit decisions."""
    from agents.agent2_director.creative_director import run as run_a2

    ctx.log("Formulating creative edit decisions...")
    ctx.set_progress(15, "Analyzing blueprint")

    edit_blueprint = ctx.inputs.get("edit_blueprint")
    fused_timeline = ctx.inputs.get("fused_timeline")

    pacing_preset = ctx.config.get("pacing_preset", "balanced")

    decisions = run_a2(
        output_dir=ctx.output_dir,
        edit_blueprint=edit_blueprint,
        fused_timeline=fused_timeline,
        pacing_preset=pacing_preset,
    )

    summary = decisions.get("editorial_summary", {})
    ctx.set_metrics({
        "keep_segments": len(decisions.get("keep_segments", [])),
        "target_duration": f"{summary.get('estimated_final_duration', 0):.1f}s",
        "cuts_applied": summary.get("total_cuts_applied", 0),
        "punch_ins": summary.get("punch_ins_count", 0),
        "b_roll_windows": summary.get("b_roll_windows_count", 0),
    })
    ctx.add_artifact("creative_decisions.json", str(ctx.output_dir / "creative_decisions.json"), "json")
    ctx.set_progress(100, "Creative decisions finalized")

    return {"creative_decisions": decisions}


def execute_agent3_assembler(ctx: ExecutionContext) -> dict[str, Any]:
    """Agent 3: Timeline Assembler, Render Compiler & Edit Validator."""
    from agents.agent3_assembler.timeline_assembler import run as run_a3

    ctx.log("Starting multi-track timeline assembly, render compilation & edit validation...")
    ctx.set_progress(10, "Composing multi-track Timeline IR")

    video_path = ctx.inputs.get("video_file", ctx.video_path)
    creative_decisions = ctx.inputs.get("creative_decisions")
    fused_timeline = ctx.inputs.get("fused_timeline")

    render_result = run_a3(
        output_dir=ctx.output_dir,
        video_path=video_path,
        creative_decisions=creative_decisions,
        fused_timeline=fused_timeline,
    )

    final_video = render_result.get("final_video")
    edl_file = render_result.get("edl_file")
    srt_file = render_result.get("srt_file")
    ass_file = render_result.get("ass_file")
    timeline_ir_file = render_result.get("timeline_ir_file")
    validation_report_file = render_result.get("validation_report_file")

    ctx.set_metrics({
        "rendered_duration": f"{render_result.get('rendered_duration', 0):.1f}s",
        "resolution": render_result.get("resolution", "1080x1920"),
        "aspect_ratio": render_result.get("aspect_ratio", "9:16"),
        "file_size": f"{render_result.get('file_size_mb', 0)} MB",
        "cuts_executed": render_result.get("cuts_executed", 0),
        "punch_ins": render_result.get("punch_ins_executed", 0),
        "captions": render_result.get("captions_compiled", 0),
        "validation": render_result.get("validation_status", "PASSED"),
        "render_time": f"{render_result.get('render_time_seconds', 0)}s",
        "status": "Ready to Download",
    })

    if final_video and Path(final_video).exists():
        ctx.add_artifact("final_reel_edited.mp4", str(final_video), "video")
    if timeline_ir_file and Path(timeline_ir_file).exists():
        ctx.add_artifact("timeline_ir.json", str(timeline_ir_file), "json")
    if validation_report_file and Path(validation_report_file).exists():
        ctx.add_artifact("validation_report.json", str(validation_report_file), "json")
    if edl_file and Path(edl_file).exists():
        ctx.add_artifact("timeline.edl", str(edl_file), "edl")
    if srt_file and Path(srt_file).exists():
        ctx.add_artifact("subtitles.srt", str(srt_file), "subtitles")
    if ass_file and Path(ass_file).exists():
        ctx.add_artifact("subtitles.ass", str(ass_file), "subtitles")

    ctx.set_progress(100, f"Final reel rendered & validated successfully ({render_result.get('validation_status')})!")

    return {
        "final_video": final_video,
        "timeline_ir": timeline_ir_file,
        "validation_report": validation_report_file,
        "edl_file": edl_file,
        "subtitles": srt_file,
        "ass_subtitles": ass_file,
    }




# ─── Executor Registry Builder ──────────────────────────────────────────────

def get_all_executors() -> dict[str, callable]:
    """Return a mapping of node_id → executor function for all implemented nodes."""
    return {
        "node_s1_media": execute_s1_media,
        "node_s2_speech": execute_s2_speech,
        "node_s3_audio": execute_s3_audio,
        "node_s4_visual": execute_s4_visual,
        "node_s5_fusion": execute_s5_fusion,
        "node_s6_story": execute_s6_story,
        "node_s7_opportunities": execute_s7_opportunities,
        "node_s8_blueprint": execute_s8_blueprint,
        "node_agent2": execute_agent2_director,
        "node_agent3": execute_agent3_assembler,
    }


