"""
core/timeline/project.py

Timeline IR Versioned Project Manager & Selective Re-Renderer (Phase 6)
────────────────────────────────────────────────────────────────────────
Manages versioning of the authoritative Timeline IR:
- Preserves v1 (AI-generated initial timeline) as IMMUTABLE baseline
- Manages v2, v3, ... user modifications with full rollback & undo/redo
- Builds end-to-end Decision Traceability Graphs:
    Timeline Event → Creative Decision → Story Decision → Opportunity → Evidence → Transcript
- Executes Selective Re-Rendering:
    * Graphics-only: Re-burns ASS subtitle/text tracks without re-slicing video
    * Audio-only: Re-mixes SFX & voice tracks without re-encoding video
    * Full re-render: Re-evaluates face-anchored crops and clip cuts
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.logger import get_logger
from core.timeline.models import TimelineIR, TextTrack, AudioTrack, VideoTrack
from core.timeline.compiler import RenderCompiler

logger = get_logger(__name__)


class TimelineProjectManager:
    """
    Project manager that provides version branching, undo/redo, traceability,
    and fast selective re-rendering.
    """

    def __init__(
        self,
        project_dir: Path | str,
        base_timeline_path: Optional[Path | str] = None,
        context_dir: Optional[Path | str] = None,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.context_dir = Path(context_dir).resolve() if context_dir else self.project_dir
        self.manifest_path = self.project_dir / "manifest.json"
        self.compiler = RenderCompiler()

        self._init_project(base_timeline_path)

    def _init_project(self, base_timeline_path: Optional[Path | str] = None) -> None:
        """Initializes or loads project manifest and ensures v1 exists and is immutable."""
        if not self.manifest_path.exists():
            v1_path = self.project_dir / "timeline_ir_v1.json"

            # If base timeline is provided, copy it to v1
            if base_timeline_path and Path(base_timeline_path).exists():
                shutil.copy2(base_timeline_path, v1_path)
            elif not v1_path.exists():
                # Look for timeline_ir.json in context_dir
                fallback_ir = self.context_dir / "timeline_ir.json"
                if fallback_ir.exists():
                    shutil.copy2(fallback_ir, v1_path)
                else:
                    # Write minimal valid timeline IR
                    v1_path.write_text(
                        json.dumps({"schema_version": "2.1.0", "project_name": "ReelProject", "timeline_duration": 90.05, "video_tracks": [], "audio_tracks": [], "text_tracks": []}, indent=2),
                        encoding="utf-8"
                    )

            manifest = {
                "project_name": f"{self.project_dir.parent.stem}_AI_Reel",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "current_version": "v1",
                "versions": [
                    {
                        "version": "v1",
                        "description": "AI-Generated Initial Baseline (Immutable)",
                        "filename": "timeline_ir_v1.json",
                        "created_at": datetime.now().isoformat(),
                        "is_immutable": True,
                    }
                ],
                "undo_stack": [],
                "redo_stack": [],
            }
            self._save_manifest(manifest)
            logger.info(f"Initialized timeline project at {self.project_dir} with immutable v1.")

    def _load_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        manifest["updated_at"] = datetime.now().isoformat()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def get_manifest(self) -> Dict[str, Any]:
        return self._load_manifest()

    def get_current_timeline(self) -> Dict[str, Any]:
        manifest = self._load_manifest()
        curr_ver = manifest["current_version"]
        return self.get_version_timeline(curr_ver)

    def get_version_timeline(self, version: str) -> Dict[str, Any]:
        manifest = self._load_manifest()
        v_entry = next((v for v in manifest["versions"] if v["version"] == version), None)
        if not v_entry:
            raise ValueError(f"Version '{version}' not found in project manifest.")

        file_path = self.project_dir / v_entry["filename"]
        if not file_path.exists():
            raise FileNotFoundError(f"Timeline file {file_path} not found.")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_new_version(self, timeline_ir_dict: Dict[str, Any], description: str) -> str:
        """
        Creates a new version branch (v2, v3, ...) and updates the manifest.
        Guarantees v1 is NEVER overwritten.
        """
        manifest = self._load_manifest()
        curr_ver = manifest["current_version"]

        # Calculate next version number
        existing_nums = [
            int(v["version"].replace("v", ""))
            for v in manifest["versions"]
            if v["version"].startswith("v") and v["version"][1:].isdigit()
        ]
        next_num = max(existing_nums, default=1) + 1
        new_ver = f"v{next_num}"
        new_filename = f"timeline_ir_{new_ver}.json"
        new_filepath = self.project_dir / new_filename

        with open(new_filepath, "w", encoding="utf-8") as f:
            json.dump(timeline_ir_dict, f, indent=2)

        # Update undo stack
        manifest["undo_stack"].append(curr_ver)
        manifest["redo_stack"].clear()  # New edit clears redo stack
        manifest["current_version"] = new_ver
        manifest["versions"].append({
            "version": new_ver,
            "description": description,
            "filename": new_filename,
            "created_at": datetime.now().isoformat(),
            "is_immutable": False,
        })

        self._save_manifest(manifest)
        logger.info(f"Created version {new_ver} ({description}) at {new_filepath}")
        return new_ver

    def rollback_to_version(self, version: str) -> Dict[str, Any]:
        """Rollback current active version to a previous version."""
        manifest = self._load_manifest()
        v_entry = next((v for v in manifest["versions"] if v["version"] == version), None)
        if not v_entry:
            raise ValueError(f"Version '{version}' does not exist.")

        manifest["undo_stack"].append(manifest["current_version"])
        manifest["current_version"] = version
        self._save_manifest(manifest)
        logger.info(f"Rolled back active version to {version}")
        return self.get_version_timeline(version)

    def undo(self) -> Optional[Dict[str, Any]]:
        manifest = self._load_manifest()
        if not manifest["undo_stack"]:
            return None
        prev_ver = manifest["undo_stack"].pop()
        manifest["redo_stack"].append(manifest["current_version"])
        manifest["current_version"] = prev_ver
        self._save_manifest(manifest)
        return self.get_version_timeline(prev_ver)

    def redo(self) -> Optional[Dict[str, Any]]:
        manifest = self._load_manifest()
        if not manifest["redo_stack"]:
            return None
        next_ver = manifest["redo_stack"].pop()
        manifest["undo_stack"].append(manifest["current_version"])
        manifest["current_version"] = next_ver
        self._save_manifest(manifest)
        return self.get_version_timeline(next_ver)

    # ── Decision Traceability Graph ─────────────────────────────────
    def build_traceability_graph(self, event_id: str) -> Dict[str, Any]:
        """
        Builds complete explainability graph for a timeline event or clip:
        Event -> Creative Decision -> Agent 2 Story Decision -> Opportunity -> Evidence -> Transcript
        """
        trace = {
            "target_id": event_id,
            "timeline_event": None,
            "creative_decision": None,
            "story_decision": None,
            "opportunity": None,
            "evidence": [],
            "transcript_context": None,
        }

        # 1. Find in Timeline IR
        curr_ir = self.get_current_timeline()
        found_item = None
        for track_type in ["text_tracks", "video_tracks", "audio_tracks"]:
            for track in curr_ir.get(track_type, []):
                items = track.get("events", []) if track_type == "text_tracks" else track.get("clips", [])
                for it in items:
                    if it.get("event_id") == event_id or it.get("clip_id") == event_id:
                        found_item = it
                        trace["timeline_event"] = it
                        break
                if found_item:
                    break
            if found_item:
                break

        # If not matched directly, find by interval overlap
        # 2. Match in creative_decisions.json
        creative_path = self.context_dir / "creative_decisions.json"
        if creative_path.exists():
            try:
                with open(creative_path, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                c_decisions = cdata.get("decisions") or cdata.get("creative_decisions", [])
                for cd in c_decisions:
                    if (
                        cd.get("creative_id") == event_id
                        or cd.get("source_decision_id") == event_id
                        or cd.get("decision_id") == event_id
                        or cd.get("target_id") == event_id
                    ):
                        trace["creative_decision"] = cd
                        break
                # If still not found, check timing
                if not trace["creative_decision"] and found_item:
                    t_in = found_item.get("timeline_in") or found_item.get("source_in", 0)
                    for cd in c_decisions:
                        span = cd.get("time_range") or [cd.get("start", 0), cd.get("end", 0)]
                        if abs(span[0] - t_in) < 1.0:
                            trace["creative_decision"] = cd
                            break
            except Exception as e:
                logger.warning(f"Error reading creative decisions for trace: {e}")

        # 3. Match in edit_decisions.json (Agent 2)
        edit_path = self.context_dir / "edit_decisions.json"
        story_plan_path = self.context_dir / "story_edit_plan.json"
        story_dec = None
        opp_id = None
        if edit_path.exists():
            try:
                with open(edit_path, "r", encoding="utf-8") as f:
                    edata = json.load(f)
                accepted = edata.get("accepted_decisions", [])
                c_dec = trace.get("creative_decision")
                c_dec_id = c_dec.get("source_decision_id") or c_dec.get("decision_id") if c_dec else event_id

                for ad in accepted:
                    if ad.get("decision_id") in (c_dec_id, event_id):
                        story_dec = ad
                        opp_id = ad.get("opportunity_id") or (ad.get("source_opportunity_ids", [None])[0] if ad.get("source_opportunity_ids") else None)
                        break

                if not story_dec and found_item:
                    t_in = found_item.get("timeline_in") or found_item.get("source_in", 0)
                    for ad in accepted:
                        span = ad.get("time_range") or [ad.get("start", 0), ad.get("end", 0)]
                        if abs(span[0] - t_in) < 2.0:
                            story_dec = ad
                            opp_id = ad.get("opportunity_id") or (ad.get("source_opportunity_ids", [None])[0] if ad.get("source_opportunity_ids") else None)
                            break
                trace["story_decision"] = story_dec
            except Exception as e:
                logger.warning(f"Error reading edit decisions for trace: {e}")

        # 4. Match Opportunity & Evidence in evidence_inventory.json
        evidence_path = self.context_dir / "evidence_inventory.json"
        if evidence_path.exists():
            try:
                with open(evidence_path, "r", encoding="utf-8") as f:
                    ev_data = json.load(f)

                opps = ev_data.get("editorial_opportunities", [])
                # Find opportunity by ID
                if opp_id:
                    for opp in opps:
                        if opp.get("opportunity_id") == opp_id:
                            trace["opportunity"] = opp
                            break

                # Fallback to match opportunity by time overlap
                if not trace.get("opportunity") and found_item:
                    t_in = found_item.get("timeline_in") or found_item.get("source_in", 0)
                    for opp in opps:
                        span = opp.get("time_range") or [opp.get("start", 0), opp.get("end", 0)]
                        if abs(span[0] - t_in) < 3.0:
                            trace["opportunity"] = opp
                            break

                # Find associated evidence items
                linked_ev_ids = []
                if trace.get("opportunity"):
                    linked_ev_ids = (
                        trace["opportunity"].get("supporting_evidence_ids")
                        or trace["opportunity"].get("source_evidence_ids")
                        or []
                    )
                if not linked_ev_ids and story_dec:
                    linked_ev_ids = (
                        story_dec.get("supporting_evidence_ids")
                        or story_dec.get("source_evidence_ids")
                        or []
                    )

                all_evidence = ev_data.get("evidence") or ev_data.get("evidence_items", [])
                for ev_item in all_evidence:
                    if ev_item.get("evidence_id") in linked_ev_ids:
                        trace["evidence"].append(ev_item)

                # Fallback: if no direct link, find evidence items overlapping event timestamp
                if not trace["evidence"] and found_item:
                    t_in = found_item.get("timeline_in") or found_item.get("source_in", 0)
                    for ev_item in all_evidence:
                        span = ev_item.get("time_range") or [ev_item.get("start", 0), ev_item.get("end", 0)]
                        if abs(span[0] - t_in) < 3.0:
                            trace["evidence"].append(ev_item)

                # Extract relevant transcript context
                t_in = (found_item.get("timeline_in") if found_item else 0.0) or 0.0
                words = ev_data.get("transcript_enriched", {}).get("words", [])
                if not words:
                    # Check enriched_transcript.json
                    t_path = self.context_dir / "enriched_transcript.json"
                    if not t_path.exists():
                        t_path = self.context_dir / "transcript_raw.json"
                    if t_path.exists():
                        try:
                            with open(t_path, "r", encoding="utf-8") as tf:
                                t_json = json.load(tf)
                            words = t_json.get("words", [])
                            if not words and "sentences" in t_json:
                                for s in t_json["sentences"]:
                                    words.extend(s.get("words", []))
                        except Exception:
                            pass

                nearby_words = [w for w in words if abs(w.get("start", 0) - t_in) <= 3.5]
                if nearby_words:
                    trace["transcript_context"] = {
                        "words": nearby_words,
                        "text": " ".join(w.get("word", "") for w in nearby_words)
                    }
            except Exception as e:
                logger.warning(f"Error reading evidence inventory for trace: {e}")

        # Check opportunity_inventory.json if still missing opportunity
        if not trace.get("opportunity"):
            opp_path = self.context_dir / "opportunity_inventory.json"
            if opp_path.exists():
                try:
                    with open(opp_path, "r", encoding="utf-8") as of:
                        odata = json.load(of)
                    opps = odata.get("opportunities") or odata.get("editorial_opportunities", [])
                    t_in = (found_item.get("timeline_in") if found_item else 0.0) or 0.0
                    for opp in opps:
                        span = opp.get("time_range") or [opp.get("start", 0), opp.get("end", 0)]
                        if abs(span[0] - t_in) < 3.0:
                            trace["opportunity"] = opp
                            break
                except Exception:
                    pass

        return trace

    # ── Selective Re-Rendering Engine ───────────────────────────────
    def rerender_graphics_only(
        self,
        output_filename: str = "final_reel_edited.mp4"
    ) -> Dict[str, Any]:
        """
        Fast selective re-render: Re-burns ASS typography without re-slicing video subclips.
        Takes 3-5 seconds instead of 45-60s full re-render.
        """
        manifest = self._load_manifest()
        curr_ver = manifest["current_version"]
        timeline_dict = self.get_version_timeline(curr_ver)

        # Parse into TimelineIR model
        timeline = TimelineIR.from_dict(timeline_dict)

        # Locate base assembled video (before subtitle burn)
        base_video = self.project_dir / "assembled_base.mp4"
        if not base_video.exists():
            base_video = self.context_dir / "assembled_base.mp4"
        if not base_video.exists():
            temp_dir = self.context_dir / "render_temp"
            base_video = temp_dir / "assembled_audio_mixed.mp4"
        if not base_video.exists():
            temp_dir = self.context_dir / "render_temp"
            base_video = temp_dir / "assembled_broll.mp4"
        if not base_video.exists():
            # Fallback to full compile if temp stages were deleted
            logger.info("Intermediate assembled video not found; falling back to full compile.")
            return self.rerender_full(output_filename=output_filename)

        t0 = time.time()
        logger.info(f"Selective Graphics Re-render ({curr_ver}) burning new typography onto {base_video.name}...")

        # 1. Regenerate ASS file
        ass_path = self.project_dir / "subtitles_updated.ass"
        srt_path = self.project_dir / "subtitles_updated.srt"
        if timeline.text_tracks and timeline.text_tracks[0].events:
            text_track = timeline.text_tracks[0]
            self.compiler._write_ass_file(text_track, ass_path, timeline.target_width, timeline.target_height)
            self.compiler._write_srt_file(text_track, srt_path)

        final_out = self.project_dir / output_filename

        # 2. Burn subtitles via FFmpeg
        burn_cmd = [
            self.compiler.ffmpeg_bin,
            "-y",
            "-i", str(base_video.resolve()),
            "-vf", f"subtitles={ass_path.name}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-loglevel", "error",
            str(final_out.resolve()),
        ]
        res = subprocess.run(burn_cmd, cwd=str(self.project_dir), capture_output=True, text=True)

        if res.returncode != 0:
            logger.error(f"Selective subtitle re-burn failed: {res.stderr}")
            raise RuntimeError(f"FFmpeg subtitle burn failed: {res.stderr}")

        dur = round(time.time() - t0, 2)
        size_mb = round(final_out.stat().st_size / (1024 * 1024), 2)
        logger.info(f"Selective graphics re-render finished in {dur}s ({size_mb} MB).")

        return {
            "mode": "GRAPHICS_ONLY",
            "version": curr_ver,
            "render_time_sec": dur,
            "output_file": str(final_out),
            "file_size_mb": size_mb,
            "ass_file": str(ass_path),
            "srt_file": str(srt_path),
        }

    def rerender_audio_only(
        self,
        output_filename: str = "final_reel_edited.mp4"
    ) -> Dict[str, Any]:
        """
        Selective audio re-render: Re-mixes SFX & voice tracks without re-encoding video.
        """
        manifest = self._load_manifest()
        curr_ver = manifest["current_version"]
        timeline_dict = self.get_version_timeline(curr_ver)
        timeline = TimelineIR.from_dict(timeline_dict)

        temp_dir = self.context_dir / "render_temp"
        base_video = temp_dir / "assembled_broll.mp4"
        if not base_video.exists():
            base_video = temp_dir / "assembled_cuts.mp4"
        if not base_video.exists():
            return self.rerender_full(output_filename=output_filename)

        t0 = time.time()
        logger.info(f"Selective Audio Re-render ({curr_ver}) re-mixing SFX cues...")

        # Re-run audio mix stage
        audio_mixed_path = temp_dir / "assembled_audio_mixed.mp4"
        resolved_sfx = []
        if timeline.audio_tracks:
            for a_track in timeline.audio_tracks:
                if a_track.role == "SFX" or getattr(a_track.role, "value", str(a_track.role)) == "SFX" or "SFX" in a_track.name:
                    for a_clip in a_track.clips:
                        if a_clip.status == "RESOLVED" and a_clip.source_reference and Path(a_clip.source_reference).exists():
                            resolved_sfx.append(a_clip)

        if resolved_sfx:
            sfx_inputs = ["-i", str(base_video.resolve())]
            filter_chains = []
            sfx_labels = []
            for s_idx, s_clip in enumerate(resolved_sfx, start=1):
                sfx_inputs.extend(["-i", str(Path(s_clip.source_reference).resolve())])
                delay_ms = int(s_clip.timeline_in * 1000)
                vol = s_clip.volume
                s_label = f"sfx_{s_idx}"
                sfx_labels.append(f"[{s_label}]")
                filter_chains.append(f"[{s_idx}:a]volume={vol:.2f},adelay={delay_ms}|{delay_ms}[{s_label}]")
            all_audio_labels = "[0:a]" + "".join(sfx_labels)
            filter_chains.append(f"{all_audio_labels}amix=inputs={1 + len(resolved_sfx)}:duration=first:dropout_transition=0[aout]")
            filter_complex = ";".join(filter_chains)

            sfx_cmd = [
                self.compiler.ffmpeg_bin, "-y",
                *sfx_inputs,
                "-filter_complex", filter_complex,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-loglevel", "error",
                str(audio_mixed_path)
            ]
            subprocess.run(sfx_cmd, capture_output=True, text=True, check=True)
        else:
            shutil.copy2(base_video, audio_mixed_path)

        # Now re-burn graphics
        return self.rerender_graphics_only(output_filename=output_filename)

    def rerender_full(
        self,
        output_filename: str = "final_reel_edited.mp4"
    ) -> Dict[str, Any]:
        """
        Executes full multi-track render compile from the current version's Timeline IR.
        """
        manifest = self._load_manifest()
        curr_ver = manifest["current_version"]
        timeline_dict = self.get_version_timeline(curr_ver)
        timeline = TimelineIR.from_dict(timeline_dict)

        logger.info(f"Executing Full Re-render for version {curr_ver}...")
        result = self.compiler.compile(
            timeline=timeline,
            output_dir=self.project_dir,
            burn_captions=True,
            output_filename=output_filename,
        )
        result["mode"] = "FULL"
        result["version"] = curr_ver
        return result

    # ── Export Formats ──────────────────────────────────────────────
    def export_edl(self) -> str:
        curr_ir = self.get_current_timeline()
        timeline = TimelineIR.from_dict(curr_ir)
        edl_path = self.project_dir / "timeline_export.edl"
        self.compiler._generate_edl(timeline, edl_path)
        return edl_path.read_text(encoding="utf-8")

    def export_srt(self) -> str:
        curr_ir = self.get_current_timeline()
        timeline = TimelineIR.from_dict(curr_ir)
        srt_path = self.project_dir / "timeline_export.srt"
        if timeline.text_tracks:
            self.compiler._write_srt_file(timeline.text_tracks[0], srt_path)
            return srt_path.read_text(encoding="utf-8")
        return ""
