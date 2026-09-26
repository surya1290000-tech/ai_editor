"""
core/timeline/compiler.py

Render Compiler
───────────────
Translates the authoritative Timeline IR into physical video renders using FFmpeg.
Compiles multi-track timeline operations:
- Per-clip face-aware crops & scales (base vs punch-in jump cuts)
- Seamless subclip concatenation via FFmpeg concat demuxer
- Kinetic stylized ASS subtitle burn-in via libass
- CMX 3600 EDL export for DaVinci Resolve NLE compatibility
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import core  # Guarantees Gyan FFmpeg on PATH
from core.timeline.models import TimelineIR, VideoClip
from core.logger import get_logger

logger = get_logger(__name__)


class RenderCompiler:
    """
    Compiles a TimelineIR model into a rendered MP4 file and NLE project files.
    """

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_bin = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"

    def compile(
        self,
        timeline: TimelineIR,
        output_dir: Path | str,
        burn_captions: bool = True,
        output_filename: str = "final_reel_edited.mp4",
    ) -> Dict[str, Any]:
        """
        Executes full rendering from TimelineIR.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        final_video_path = output_dir / output_filename

        t0 = time.time()
        logger.info(
            f"RenderCompiler | Compiling timeline for '{timeline.project_name}' "
            f"({timeline.target_width}x{timeline.target_height}, {timeline.timeline_duration:.2f}s, "
            f"{len(timeline.video_tracks[0].clips) if timeline.video_tracks else 0} clips)..."
        )

        if not timeline.video_tracks or not timeline.video_tracks[0].clips:
            raise ValueError("TimelineIR contains no video clips to render")

        primary_track = timeline.video_tracks[0]
        clips = primary_track.clips
        temp_dir = output_dir / "_render_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        part_files: List[Path] = []
        total_clip_duration = 0.0

        try:
            # ── 1. Render Each Video Clip with Accurate Crop/Scale ──────
            for idx, clip in enumerate(clips):
                dur = round(clip.source_out - clip.source_in, 4)
                if dur <= 0.05:
                    continue

                part_file = temp_dir / f"clip_{idx:04d}.mp4"

                # Construct video filter from clip crop and timeline target resolution
                if clip.crop:
                    # Support both normalized (0..1) and pixel coordinates
                    if clip.crop.w <= 1.0:
                        crop_expr = (
                            f"crop=w=iw*{clip.crop.w:.4f}:h=ih*{clip.crop.h:.4f}:"
                            f"x=iw*{clip.crop.x:.4f}:y=ih*{clip.crop.y:.4f}"
                        )
                    else:
                        crop_expr = (
                            f"crop=w={int(clip.crop.w)}:h={int(clip.crop.h)}:"
                            f"x={int(clip.crop.x)}:y={int(clip.crop.y)}"
                        )
                    vf = f"{crop_expr},scale={timeline.target_width}:{timeline.target_height}:flags=bicubic,setsar=1"
                else:
                    # Default center crop to target aspect ratio
                    vf = (
                        f"crop=(ih*{timeline.target_width}/{timeline.target_height}):ih:(iw-out_w)/2:0,"
                        f"scale={timeline.target_width}:{timeline.target_height}:flags=bicubic,"
                        f"setsar=1"
                    )

                cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-ss", f"{clip.source_in:.3f}",
                    "-to", f"{clip.source_out:.3f}",
                    "-i", str(clip.source_reference),
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-loglevel", "error",
                    str(part_file),
                ]

                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if res.returncode == 0 and part_file.exists() and part_file.stat().st_size > 0:
                    part_files.append(part_file)
                    total_clip_duration += dur
                else:
                    logger.error(
                        f"RenderCompiler | Subclip {clip.clip_id} failed (code {res.returncode}): {res.stderr.strip()}"
                    )
                    raise RuntimeError(f"Failed to render subclip {clip.clip_id}: {res.stderr.strip()}")

            if not part_files:
                raise RuntimeError("No valid subclips were rendered")

            # ── 2. Seamless Concatenation via Concat Demuxer ───────────
            concat_list_path = temp_dir / "concat_list.txt"
            with open(concat_list_path, "w", encoding="utf-8") as f:
                for p in part_files:
                    clean_path = str(p.resolve()).replace("\\", "/")
                    f.write(f"file '{clean_path}'\n")

            assembled_raw_path = temp_dir / "assembled_raw.mp4"
            concat_cmd = [
                self.ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c", "copy",
                "-loglevel", "error",
                str(assembled_raw_path),
            ]
            concat_res = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=180)
            if concat_res.returncode != 0 or not assembled_raw_path.exists():
                raise RuntimeError(f"FFmpeg concat failed: {concat_res.stderr.strip()}")

            current_video_stage = assembled_raw_path

            # ── 2B. Multi-Track B-Roll Video Overlays ──────────────────
            resolved_broll: List[Any] = []
            if timeline.broll_tracks:
                for b_track in timeline.broll_tracks:
                    for b_clip in b_track.clips:
                        if b_clip.status == "RESOLVED" and b_clip.source_reference and Path(b_clip.source_reference).exists():
                            resolved_broll.append(b_clip)

            if resolved_broll:
                logger.info(f"RenderCompiler | Applying {len(resolved_broll)} resolved B-roll cutaway overlay(s)...")
                broll_out_path = temp_dir / "assembled_broll.mp4"

                broll_inputs = ["-i", str(current_video_stage.resolve())]
                filter_chains = []
                last_v = "0:v"

                for b_idx, b_clip in enumerate(resolved_broll, start=1):
                    broll_inputs.extend(["-i", str(Path(b_clip.source_reference).resolve())])
                    b_dur = round(b_clip.timeline_out - b_clip.timeline_in, 3)
                    fade_dur = min(0.3, b_dur / 3.0)
                    out_v = f"vb_{b_idx}"
                    b_label = f"broll_{b_idx}"

                    filter_chains.append(
                        f"[{b_idx}:v]scale={timeline.target_width}:{timeline.target_height}:force_original_aspect_ratio=increase,"
                        f"crop={timeline.target_width}:{timeline.target_height},setsar=1,"
                        f"fade=t=in:st=0:d={fade_dur:.2f}:alpha=1,fade=t=out:st={b_dur - fade_dur:.2f}:d={fade_dur:.2f}:alpha=1[{b_label}]"
                    )
                    filter_chains.append(
                        f"[{last_v}][{b_label}]overlay=0:0:enable='between(t,{b_clip.timeline_in:.3f},{b_clip.timeline_out:.3f})'[{out_v}]"
                    )
                    last_v = out_v

                filter_complex = ";".join(filter_chains)
                broll_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    *broll_inputs,
                    "-filter_complex", filter_complex,
                    "-map", f"[{last_v}]",
                    "-map", "0:a",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                    "-c:a", "copy",
                    "-loglevel", "error",
                    str(broll_out_path),
                ]
                broll_res = subprocess.run(broll_cmd, capture_output=True, text=True, timeout=240)
                if broll_res.returncode == 0 and broll_out_path.exists():
                    current_video_stage = broll_out_path
                    logger.info(f"RenderCompiler | B-roll overlays successfully rendered.")
                else:
                    logger.warning(f"RenderCompiler | B-roll overlay failed ({broll_res.stderr.strip()}), proceeding with base video.")

            # ── 2C. Multi-Track SFX Audio Mixing ───────────────────────
            resolved_sfx: List[Any] = []
            if timeline.audio_tracks:
                for a_track in timeline.audio_tracks:
                    if a_track.role == "SFX" or getattr(a_track.role, "value", str(a_track.role)) == "SFX" or "SFX" in a_track.name:
                        for a_clip in a_track.clips:
                            if a_clip.status == "RESOLVED" and a_clip.source_reference and Path(a_clip.source_reference).exists():
                                resolved_sfx.append(a_clip)

            if resolved_sfx:
                logger.info(f"RenderCompiler | Mixing {len(resolved_sfx)} SFX audio cue(s) into master audio...")
                audio_out_path = temp_dir / "assembled_audio_mixed.mp4"

                sfx_inputs = ["-i", str(current_video_stage.resolve())]
                filter_chains = []
                sfx_labels = []

                for s_idx, s_clip in enumerate(resolved_sfx, start=1):
                    sfx_inputs.extend(["-i", str(Path(s_clip.source_reference).resolve())])
                    delay_ms = int(s_clip.timeline_in * 1000)
                    vol = s_clip.volume
                    s_label = f"sfx_{s_idx}"
                    sfx_labels.append(f"[{s_label}]")
                    filter_chains.append(
                        f"[{s_idx}:a]volume={vol:.2f},adelay={delay_ms}|{delay_ms}[{s_label}]"
                    )

                all_audio_labels = "[0:a]" + "".join(sfx_labels)
                total_audio_inputs = 1 + len(resolved_sfx)
                filter_chains.append(
                    f"{all_audio_labels}amix=inputs={total_audio_inputs}:duration=first:dropout_transition=0[aout]"
                )

                filter_complex = ";".join(filter_chains)
                sfx_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    *sfx_inputs,
                    "-filter_complex", filter_complex,
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-loglevel", "error",
                    str(audio_out_path),
                ]
                sfx_res = subprocess.run(sfx_cmd, capture_output=True, text=True, timeout=180)
                if sfx_res.returncode == 0 and audio_out_path.exists():
                    current_video_stage = audio_out_path
                    logger.info("RenderCompiler | SFX cues successfully mixed with master voice track.")
                else:
                    logger.warning(f"RenderCompiler | SFX mixing failed ({sfx_res.stderr.strip()}), proceeding with master audio.")

            # ── 3. Burn-in Kinetic Subtitles & Hero Text Cards ─────────
            ass_path = output_dir / "subtitles.ass"
            srt_path = output_dir / "subtitles.srt"

            if timeline.text_tracks and timeline.text_tracks[0].events:
                text_track = timeline.text_tracks[0]
                self._write_ass_file(text_track, ass_path, timeline.target_width, timeline.target_height)
                self._write_srt_file(text_track, srt_path)

            # Save assembled base video for fast selective re-rendering
            assembled_base_path = output_dir / "assembled_base.mp4"
            try:
                shutil.copy2(current_video_stage, assembled_base_path)
            except Exception as e:
                logger.warning(f"Could not save assembled_base.mp4: {e}")

            if burn_captions and ass_path.exists() and ass_path.stat().st_size > 0:
                logger.info(f"RenderCompiler | Burning kinetic subtitles from '{ass_path.name}' into final export...")
                burn_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-i", str(current_video_stage.resolve()),
                    "-vf", f"subtitles={ass_path.name}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    "-loglevel", "error",
                    str(final_video_path.resolve()),
                ]
                burn_res = subprocess.run(
                    burn_cmd,
                    cwd=str(output_dir),
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                if burn_res.returncode != 0:
                    logger.warning(
                        f"RenderCompiler | Subtitle burn failed ({burn_res.stderr.strip()}), falling back to assembled video"
                    )
                    shutil.copy2(current_video_stage, final_video_path)
            else:
                shutil.copy2(current_video_stage, final_video_path)

            # ── 4. Generate CMX 3600 EDL for DaVinci Resolve ───────────
            edl_path = output_dir / "timeline.edl"
            self._generate_edl(timeline, edl_path)

            render_time = round(time.time() - t0, 2)
            file_size_mb = round(final_video_path.stat().st_size / (1024 * 1024), 2) if final_video_path.exists() else 0.0

            logger.info(
                f"RenderCompiler | Export complete: '{final_video_path.name}' ({file_size_mb} MB) "
                f"rendered in {render_time}s"
            )

            return {
                "final_video": str(final_video_path),
                "edl_file": str(edl_path) if edl_path.exists() else None,
                "srt_file": str(srt_path) if srt_path.exists() else None,
                "ass_file": str(ass_path) if ass_path.exists() else None,
                "file_size_mb": file_size_mb,
                "render_time_seconds": render_time,
                "clips_rendered": len(part_files),
                "punch_ins_rendered": sum(1 for c in clips if c.is_punch_in),
                "broll_rendered": len(resolved_broll),
                "sfx_rendered": len(resolved_sfx),
                "timeline_duration": timeline.timeline_duration,
            }

        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    def _write_ass_file(self, text_track, ass_path: Path, width: int, height: int):
        """Writes stylized ASS subtitle script with dual styles for HeroText and Kinetic captions."""
        import re

        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ReelKinetic,Arial Black,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,50,50,380,1
Style: ReelHero,Montserrat,84,&H00F5F5F5,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,1,5,3,5,60,60,0,1
Style: ReelTelugu,Nirmala UI,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,50,50,380,1
Style: ReelTeluguHero,Nirmala UI,84,&H00F5F5F5,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,1,5,3,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events_lines = []
        for ev in text_track.events:
            start_tc = self._seconds_to_ass_time(ev.timeline_in)
            end_tc = self._seconds_to_ass_time(ev.timeline_out)

            is_indic = bool(re.search(r"[\u0c00-\u0c7f]", ev.text))

            # Style choice based on position_align
            if ev.position_align == 5:
                # Hero Text center title card
                style_name = "ReelTeluguHero" if is_indic else "ReelHero"
                if "{\\" in ev.text:
                    line_text = ev.text
                else:
                    line_text = f"{{\\c&H007BD4F6&}}{ev.text.upper()}{{\\c&H00F5F5F5&}}"
            else:
                # Kinetic caption lower third
                style_name = "ReelTelugu" if is_indic else "ReelKinetic"
                if "{\\" in ev.text:
                    line_text = ev.text
                else:
                    words = ev.text.split()
                    formatted = []
                    for w in words:
                        clean = w.strip(".,!?\"'").lower()
                        if clean in [tok.lower() for tok in ev.emphasis_tokens]:
                            formatted.append(f"{{\\c&H0000D7FF&}}{w.upper()}{{\\c&H00FFFFFF&}}")
                        else:
                            formatted.append(w.upper())
                    line_text = " ".join(formatted)

            events_lines.append(f"Dialogue: 0,{start_tc},{end_tc},{style_name},,0,0,0,,{line_text}")

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_header + "\n".join(events_lines))

    def _write_srt_file(self, text_track, srt_path: Path):
        """Writes standard SRT subtitle file."""
        import re
        lines = []
        for idx, ev in enumerate(text_track.events, start=1):
            start_tc = self._seconds_to_srt_time(ev.timeline_in)
            end_tc = self._seconds_to_srt_time(ev.timeline_out)
            # Strip any ASS tags for plain SRT
            clean_text = re.sub(r"\{.*?\}", "", ev.text).strip()
            lines.append(f"{idx}")
            lines.append(f"{start_tc} --> {end_tc}")
            lines.append(clean_text)
            lines.append("")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


    def _generate_edl(self, timeline: TimelineIR, edl_path: Path):
        """Generates CMX 3600 EDL for DaVinci Resolve Free."""
        source_name = Path(timeline.source_video).name
        header = [
            f"TITLE: {timeline.project_name}",
            "FCM: NON-DROP FRAME",
            "",
        ]

        if not timeline.video_tracks:
            return

        fps = timeline.fps
        for idx, clip in enumerate(timeline.video_tracks[0].clips, start=1):
            src_in_tc = self._seconds_to_smpte(clip.source_in, fps)
            src_out_tc = self._seconds_to_smpte(clip.source_out, fps)
            rec_in_tc = self._seconds_to_smpte(clip.timeline_in, fps)
            rec_out_tc = self._seconds_to_smpte(clip.timeline_out, fps)

            header.append(f"{idx:03d}  AX       V     C        {src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}")
            header.append(f"* FROM CLIP NAME: {source_name}")
            if clip.is_punch_in:
                header.append(f"* COMMENT: PUNCH_IN_SCALE_{clip.scale:.2f}")
            header.append(f"{idx:03d}  AX       A     C        {src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}")
            header.append(f"* FROM CLIP NAME: {source_name}")
            header.append("")

        with open(edl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header))

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _seconds_to_ass_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

    @staticmethod
    def _seconds_to_smpte(seconds: float, fps: float = 30.0) -> str:
        total_frames = int(round(seconds * fps))
        frames = total_frames % int(fps)
        total_sec = total_frames // int(fps)
        s = total_sec % 60
        m = (total_sec // 60) % 60
        h = (total_sec // 3600) % 24
        return f"{h:02d}:{m:02d}:{s:02d}:{frames:02d}"
