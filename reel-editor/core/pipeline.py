"""
core/pipeline.py

Unified Production Execution Pipeline for AI Reel Editor
────────────────────────────────────────────────────────
Executes the complete end-to-end workflow on ANY input video:
  INPUT VIDEO
  → PERCEPTION (Media extraction, ASR, Smart Visual Sampling, Audio features, Feature fusion)
  → AGENT 1 OBSERVER (Multimodal evidence synthesis, narrative acts, opportunity catalogue)
  → AGENT 2 STORY EDITOR (Selective story editing, budget tracking, NO_EDIT preservation)
  → AGENT 3 CREATIVE DIRECTOR (Style profile, typography, face-aware punch framing, asset resolution)
  → TIMELINE IR (Authoritative multi-track representation)
  → RENDER COMPILER (FFmpeg multi-track render: cuts, punch-ins, B-roll overlays, SFX mix, ASS subtitles)
  → CRITIC / QA PASS (Forensic validation comparing decisions vs physical render)
  → HUMAN-QUALITY EDITORIAL REVIEW (Side-by-side frame captures, density metrics, 5 rubric scores)
  → OUTPUT (Final reel MP4, CMX 3600 EDL, SRT subtitles, interactive review report, versioned project)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import core
from core.logger import get_logger
import uuid
from core.cache import StageCacheManager, PIPELINE_VERSION, compute_file_sha256, compute_config_hash
from core.hardware_detector import detect_hardware
from core.model_selector import select_models
from core.evidence.smart_sampler import SmartVisualSampler
from core.evidence.pipeline import run_agent1_observer_pipeline
from agents.agent2_director.story_editor import StoryEditor
from agents.agent2_director.models import StoryEditPlan, EditDecisionsDocument
from agents.agent2_director.reporter import generate_editorial_html_report
from agents.agent3_creative.creative_director import CreativeDirector
from agents.agent3_creative.timeline_compiler import CreativeTimelineCompiler
from agents.agent3_creative.reporter import CreativeReporter
from core.timeline.models import TimelineIR
from core.timeline.compiler import RenderCompiler
from core.timeline.critic_qa import CriticQAEvaluator
from core.review.reviewer import EditorialReviewer
from core.review.reporter import EditorialQualityReporter
from core.timeline.project import TimelineProjectManager

logger = get_logger("production_pipeline")


class ProductionPipeline:
    """
    End-to-end automated pipeline executing the complete AI Reel Editor stack.
    """

    def __init__(self, workspace_root: Optional[Path | str] = None):
        self.workspace_root = Path(workspace_root or Path(__file__).resolve().parent.parent)

    def run(
        self,
        video_path: Path | str,
        output_dir: Optional[Path | str] = None,
        style_preset: str = "EDITORIAL_CINEMATIC",
        pacing_preset: str = "balanced",
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the complete pipeline autonomously on the specified input video.
        """
        v_path = Path(video_path).resolve()
        if not v_path.exists():
            raise FileNotFoundError(f"Input video not found: {v_path}")

        # Run identity and cryptographic provenance
        source_sha256 = compute_file_sha256(v_path)
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        cache_mgr = StageCacheManager(pipeline_version=PIPELINE_VERSION)

        # Setup isolated output directory
        if output_dir:
            out_dir = Path(output_dir).resolve()
        else:
            out_dir = self.workspace_root / "outputs" / f"{v_path.stem}_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Record run manifest
        run_manifest = {
            "run_id": run_id,
            "source_video_name": v_path.name,
            "source_video_sha256": source_sha256,
            "pipeline_version": PIPELINE_VERSION,
            "started_at": datetime.now().isoformat(),
            "style_preset": style_preset,
            "pacing_preset": pacing_preset,
        }
        with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2, sort_keys=True)

        pipeline_start_time = time.time()
        stage_timings = {}

        logger.info("==================================================================")
        logger.info(f"AI REEL EDITOR — PRODUCTION PIPELINE (v{PIPELINE_VERSION})")
        logger.info(f"Run ID:       {run_id}")
        logger.info(f"Input Video:  {v_path.name} (SHA: {source_sha256[:12]}...)")
        logger.info(f"Output Dir:   {out_dir}")
        logger.info(f"Style:        {style_preset} | Pacing: {pacing_preset}")
        logger.info("==================================================================")

        # ── 1. Perception Layer ─────────────────────────────────────────
        hw_profile = detect_hardware()
        hw_cache = self.workspace_root / "outputs" / "hardware_profile.json"
        models = {}
        if hw_cache.exists():
            try:
                with open(hw_cache, "r", encoding="utf-8") as f:
                    hw_data = json.load(f)
                models = hw_data.get("models", {})
            except Exception:
                models = {}
        if not models:
            deps = {"ffmpeg_available": True, "ollama_running": True, "ollama_models": []}
            models = select_models(hw_profile, deps)

        # Stage 1: Media Extraction (with SHA-256 cache verification)
        t0 = time.time()
        if cache_mgr.is_stage_valid(out_dir, "media_extraction", ["media_meta.json", "audio.wav"], v_path):
            logger.info("Stage 1 | Valid cache found for media extraction. Reusing artifacts.")
            with open(out_dir / "media_meta.json", "r", encoding="utf-8") as f:
                media_meta = json.load(f)
            stage_timings["media_extraction"] = 0.01
        else:
            from agents.agent1_analyzer.stages.s1_media_extraction import run as run_s1
            media_meta = run_s1(
                video_path=v_path,
                output_dir=out_dir,
                frame_sample_fps=models.get("frame_sample_fps", 1),
            )
            cache_mgr.write_stage_manifest(out_dir, "media_extraction", v_path)
            stage_timings["media_extraction"] = round(time.time() - t0, 2)

        # Stage 2: Multilingual Speech Extraction (Whisper int8 with model-bound cache)
        t0 = time.time()
        whisper_model_id = models.get("whisper_model", "base")
        if cache_mgr.is_stage_valid(out_dir, "speech_extraction", ["transcript_raw.json"], v_path, model_version=whisper_model_id):
            logger.info("Stage 2 | Valid cache found for speech extraction. Reusing transcript.")
            with open(out_dir / "transcript_raw.json", "r", encoding="utf-8") as f:
                transcript_raw = json.load(f)
            stage_timings["speech_extraction"] = 0.01
        else:
            from agents.agent1_analyzer.stages.s2_speech_extraction import run as run_s2
            transcript_raw = run_s2(
                audio_path=Path(media_meta["audio_path"]),
                output_dir=out_dir,
                whisper_model=models["whisper_model"],
                whisper_device=models["whisper_device"],
                whisper_compute_type=models["whisper_compute_type"],
                language_hint=None,
            )
            cache_mgr.write_stage_manifest(out_dir, "speech_extraction", v_path, model_version=whisper_model_id)
            stage_timings["speech_extraction"] = round(time.time() - t0, 2)

        # Stage 3: Audio Signal Features
        t0 = time.time()
        if cache_mgr.is_stage_valid(out_dir, "audio_features", ["audio_features.json"], v_path):
            logger.info("Stage 3 | Valid cache found for audio features. Reusing artifacts.")
            with open(out_dir / "audio_features.json", "r", encoding="utf-8") as f:
                audio_features = json.load(f)
            stage_timings["audio_features"] = 0.01
        else:
            from agents.agent1_analyzer.stages.s3_audio_features import run as run_s3
            audio_features = run_s3(
                audio_path=Path(media_meta["audio_path"]),
                transcript=transcript_raw,
                output_dir=out_dir,
            )
            cache_mgr.write_stage_manifest(out_dir, "audio_features", v_path)
            stage_timings["audio_features"] = round(time.time() - t0, 2)

        # Stage 4: Visual CV & Face Tracking
        t0 = time.time()
        if cache_mgr.is_stage_valid(out_dir, "visual_cv", ["visual_features.json"], v_path):
            logger.info("Stage 4 | Valid cache found for visual CV. Reusing artifacts.")
            with open(out_dir / "visual_features.json", "r", encoding="utf-8") as f:
                visual_features = json.load(f)
            stage_timings["visual_cv"] = 0.01
        else:
            from agents.agent1_analyzer.stages.s4_visual_cv import run as run_s4
            visual_features = run_s4(
                frames_dir=out_dir / "frames",
                output_dir=out_dir,
                media_meta=media_meta,
                mediapipe_complexity=0,
            )
            cache_mgr.write_stage_manifest(out_dir, "visual_cv", v_path)
            stage_timings["visual_cv"] = round(time.time() - t0, 2)

        # Stage 4B: Smart Visual Sampling (Waveform Filmstrips + Micro-Bursts)
        t0 = time.time()
        sampler_out = out_dir / "smart_visual_samples"
        if cache_mgr.is_stage_valid(sampler_out, "smart_sampling", ["smart_visual_manifest.json"], v_path):
            logger.info("Stage 4B | Valid cache found for smart visual sampling. Reusing manifest.")
            with open(sampler_out / "smart_visual_manifest.json", "r", encoding="utf-8") as f:
                smart_visual_manifest = json.load(f)
            stage_timings["smart_visual_sampling"] = 0.01
        else:
            logger.info("Perception | Running Smart Visual Sampling (Waveform Filmstrips & Micro-Bursts)...")
            sampler = SmartVisualSampler(base_cadence_sec=5.0, max_reasoning_frames=20)
            smart_visual_manifest = sampler.sample_video(
                video_path=v_path,
                output_dir=out_dir,
                audio_features=audio_features,
                transcript_raw=transcript_raw,
                media_meta=media_meta,
                visual_features=visual_features,
            )
            cache_mgr.write_stage_manifest(sampler_out, "smart_sampling", v_path)
            stage_timings["smart_visual_sampling"] = round(time.time() - t0, 2)

        # Stage 5: Feature Fusion
        t0 = time.time()
        from agents.agent1_analyzer.stages.s5_feature_fusion import run as run_s5
        fused_timeline = run_s5(
            output_dir=out_dir,
            media_meta=media_meta,
            transcript=transcript_raw,
            audio_features=audio_features,
            visual_features=visual_features,
        )
        stage_timings["feature_fusion"] = round(time.time() - t0, 2)

        # ── 2. Agent 1 Observer: Standardized Evidence Inventory ─────────
        logger.info("\n>>> [2/7] AGENT 1: OBSERVER & EVIDENCE SYNTHESIS")
        t0 = time.time()
        obs_results = run_agent1_observer_pipeline(
            output_dir=out_dir,
            media_meta=media_meta,
            transcript_raw=transcript_raw,
            audio_features=audio_features,
            visual_features=visual_features,
            fused_timeline=fused_timeline,
        )
        stage_timings["agent1_observer"] = round(time.time() - t0, 2)

        # ── 3. Agent 2 Story Editor: Editorial Decisions ────────────────
        logger.info("\n>>> [3/7] AGENT 2: TRUE STORY EDITOR")
        t0 = time.time()
        story_editor = StoryEditor(
            evidence_inv=obs_results["evidence_inventory"],
            story_obs=obs_results["story_observations"],
            opp_inv=obs_results["opportunity_inventory"],
            transcript=obs_results["enriched_transcript"],
            media_meta=media_meta,
            pacing_preset=pacing_preset,
            budget_config_path=self.workspace_root / "config" / "edit_budget.yaml",
        )
        agent2_docs = story_editor.run_editorial_pipeline()
        story_edit_plan = agent2_docs["story_edit_plan"]
        edit_decisions_doc = agent2_docs["edit_decisions"]

        # Persist Agent 2 deliverables
        with open(out_dir / "story_edit_plan.json", "w", encoding="utf-8") as f:
            json.dump(story_edit_plan.to_dict(), f, indent=2)
        with open(out_dir / "edit_decisions.json", "w", encoding="utf-8") as f:
            json.dump(edit_decisions_doc.to_dict(), f, indent=2)

        generate_editorial_html_report(
            story_plan=story_edit_plan,
            decisions_doc=edit_decisions_doc,
            transcript=obs_results["enriched_transcript"],
            evidence_inv=obs_results["evidence_inventory"],
            output_path=out_dir / "editorial_report.html",
        )
        stage_timings["agent2_story_editor"] = round(time.time() - t0, 2)

        # ── 4. Agent 3 Creative Director: Multi-Track Timeline IR ────────
        logger.info("\n>>> [4/7] AGENT 3: CREATIVE DIRECTOR & ASSET RESOLUTION")
        t0 = time.time()
        director = CreativeDirector(
            style_profile=style_preset,
            workspace_root=self.workspace_root,
        )
        creative_plan = director.direct(
            edit_decisions_data=edit_decisions_doc.to_dict(),
            story_edit_plan_data=story_edit_plan.to_dict(),
            media_meta=media_meta,
            visual_features=visual_features,
            transcript_raw=transcript_raw,
        )

        with open(out_dir / "creative_decisions.json", "w", encoding="utf-8") as f:
            json.dump(creative_plan.to_dict(), f, indent=2)

        # Compile into Timeline IR
        compiler_ir = CreativeTimelineCompiler()
        timeline_ir, reframe_plan = compiler_ir.compile(
            creative_plan=creative_plan,
            source_video_path=v_path,
            media_meta=media_meta,
            transcript_raw=transcript_raw,
            fused_timeline=fused_timeline,
        )
        with open(out_dir / "timeline_ir.json", "w", encoding="utf-8") as f:
            json.dump(timeline_ir.to_dict(), f, indent=2)

        # Creative report
        c_reporter = CreativeReporter()
        c_reporter.generate(
            creative_plan=creative_plan,
            edit_decisions=edit_decisions_doc.to_dict(),
            output_path=out_dir / "creative_report.html",
        )
        stage_timings["agent3_creative_director"] = round(time.time() - t0, 2)

        # ── 5. Render Compiler: Physical Video Render ────────────────────
        logger.info("\n>>> [5/7] RENDER COMPILER (FFmpeg Multi-Track)")
        t0 = time.time()
        render_compiler = RenderCompiler()
        render_result = render_compiler.compile(
            timeline=timeline_ir,
            output_dir=out_dir,
            burn_captions=True,
            output_filename="final_reel_edited.mp4",
        )
        stage_timings["render_compiler"] = round(time.time() - t0, 2)

        # ── 6. Critic / QA Pass ──────────────────────────────────────────
        logger.info("\n>>> [6/7] FORENSIC CRITIC / QA PASS")
        t0 = time.time()
        critic = CriticQAEvaluator()
        critic_result = critic.evaluate(
            rendered_video_path=render_result["final_video"],
            timeline_ir=timeline_ir,
            edit_decisions=edit_decisions_doc.to_dict(),
            creative_plan=creative_plan.to_dict(),
            output_dir=out_dir,
        )
        stage_timings["critic_qa"] = round(time.time() - t0, 2)

        # ── 7. Human-Quality Review & Editable Project Init ──────────────
        logger.info("\n>>> [7/7] HUMAN-QUALITY REVIEW & VERSIONED PROJECT INIT")
        t0 = time.time()
        reviewer = EditorialReviewer()
        review_data = reviewer.review(
            raw_video_path=v_path,
            edited_video_path=render_result["final_video"],
            edit_decisions=edit_decisions_doc.to_dict(),
            creative_plan=creative_plan.to_dict(),
            timeline_ir=timeline_ir.to_dict(),
            output_dir=out_dir,
        )

        reporter = EditorialQualityReporter()
        html_report_path = out_dir / "human_quality_review" / "human_quality_report.html"
        reporter.generate_html_report(
            report_data=review_data,
            output_file=html_report_path,
            embed_images=True,
        )

        # Initialize versioned project manager
        project_dir = out_dir / "project"
        proj_mgr = TimelineProjectManager(
            project_dir=project_dir,
            base_timeline_path=out_dir / "timeline_ir.json",
            context_dir=out_dir,
        )
        stage_timings["human_review_and_project"] = round(time.time() - t0, 2)

        total_runtime = round(time.time() - pipeline_start_time, 2)
        logger.info(f"\nPipeline successfully completed in {total_runtime}s!")

        return {
            "source_video": str(v_path.name),
            "output_dir": str(out_dir),
            "duration_seconds": media_meta.get("duration_seconds", 0),
            "runtime_seconds": total_runtime,
            "stage_timings": stage_timings,
            "asr": {
                "detected_language": transcript_raw.get("language"),
                "total_words": len(transcript_raw.get("words", [])),
                "has_code_switching": transcript_raw.get("has_code_switching", False),
            },
            "smart_sampling": {
                "total_keyframes": smart_visual_manifest.get("total_sampled_keyframes", 0),
                "total_composites": smart_visual_manifest.get("total_composites_generated", 0),
            },
            "agent1_evidence": {
                "total_evidence": len(obs_results["evidence_inventory"].evidence),
                "total_opportunities": len(obs_results["opportunity_inventory"].opportunities),
            },
            "agent2_decisions": {
                "accepted": len(edit_decisions_doc.accepted_decisions),
                "no_edit_spans": len(edit_decisions_doc.no_edit_decisions),
                "rejected": len(edit_decisions_doc.rejected_decisions),
            },
            "agent3_creative": {
                "style_profile": creative_plan.style_profile,
                "total_decisions": creative_plan.total_decisions,
                "resolved_assets": creative_plan.resolved_count,
                "unresolved_assets": creative_plan.unresolved_assets_count,
            },
            "timeline": {
                "duration": timeline_ir.timeline_duration,
                "clips_count": len(timeline_ir.video_tracks[0].clips) if timeline_ir.video_tracks else 0,
                "text_events_count": len(timeline_ir.text_tracks[0].events) if timeline_ir.text_tracks else 0,
            },
            "render": {
                "final_video": render_result["final_video"],
                "file_size_mb": render_result.get("file_size_mb", 0),
            },
            "critic_qa": {
                "verdict": critic_result.get("status", "FAIL"),
                "overall_pass": critic_result.get("overall_pass", False),
                "total_checks": critic_result.get("summary", {}).get("total_operations_verified", 0),
                "passed_checks": critic_result.get("summary", {}).get("passed_operations", 0),
                "failed_checks": critic_result.get("summary", {}).get("failed_operations", 0),
            },
            "human_review": {
                "report_html": str(html_report_path),
                "pacing_classification": review_data["density_metrics"]["pacing_classification"],
                "major_edits_per_minute": review_data["density_metrics"]["major_edits_per_minute"],
                "overall_story_value": review_data["quality_scores"]["overall_story_value"],
                "overall_naturalness_score": review_data["quality_scores"]["overall_naturalness_score"],
            },
        }
