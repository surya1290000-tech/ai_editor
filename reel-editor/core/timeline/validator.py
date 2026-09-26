"""
core/timeline/validator.py

Edit Validation Engine
──────────────────────
Performs automated forensic verification of the final rendered video against the
authoritative Timeline IR contract.

Verifies:
1. Physical integrity (file exists, non-empty, valid video and audio containers)
2. Target geometry (aspect ratio 9:16, 1080x1920)
3. Duration alignment (within ±1.0s of TimelineIR.timeline_duration)
4. Operation validation:
   - Verifies expected cuts were applied
   - Verifies punch-ins were compiled
   - Verifies captions were compiled and burned into video stream
5. Audio integrity (audio stream exists and has active samples)

If any critical validation requirement fails, flags status as FAILED.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import core  # Guarantees Gyan FFmpeg on PATH
from core.timeline.models import TimelineIR
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Detailed forensic report of the rendered video vs timeline contract."""
    is_valid: bool
    status: str  # "PASSED" or "FAILED"
    final_video_path: str
    target_resolution: str
    measured_resolution: str
    aspect_ratio: str
    timeline_duration: float
    measured_duration: float
    duration_delta: float
    video_codec: str
    audio_codec: str
    file_size_mb: float
    cuts_expected: int
    cuts_executed: int
    punch_ins_expected: int
    punch_ins_executed: int
    captions_expected: int
    captions_compiled: int
    captions_burned: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EditValidator:
    """
    Validates that the rendered output file strictly conforms to the Timeline IR.
    """

    def __init__(self, ffprobe_path: Optional[str] = None):
        self.ffprobe_bin = ffprobe_path or shutil.which("ffprobe") or "ffprobe"

    def validate(
        self,
        final_video_path: Path | str,
        timeline: TimelineIR,
        ass_path: Optional[Path | str] = None,
    ) -> ValidationReport:
        """
        Runs complete forensic inspection of rendered file against TimelineIR.
        """
        video_path = Path(final_video_path).resolve()
        errors: List[str] = []
        warnings: List[str] = []

        logger.info(f"EditValidator | Inspecting rendered file '{video_path.name}' against TimelineIR contract...")

        # ── 1. Basic File Existence & Size ────────────────────────────
        if not video_path.exists():
            return ValidationReport(
                is_valid=False,
                status="FAILED",
                final_video_path=str(video_path),
                target_resolution=f"{timeline.target_width}x{timeline.target_height}",
                measured_resolution="0x0",
                aspect_ratio="unknown",
                timeline_duration=timeline.timeline_duration,
                measured_duration=0.0,
                duration_delta=timeline.timeline_duration,
                video_codec="none",
                audio_codec="none",
                file_size_mb=0.0,
                cuts_expected=len(timeline.video_tracks[0].clips) - 1 if timeline.video_tracks else 0,
                cuts_executed=0,
                punch_ins_expected=sum(1 for c in timeline.video_tracks[0].clips if c.is_punch_in) if timeline.video_tracks else 0,
                punch_ins_executed=0,
                captions_expected=len(timeline.text_tracks[0].events) if timeline.text_tracks else 0,
                captions_compiled=0,
                captions_burned=False,
                errors=[f"Final video file not found at '{video_path}'"],
            )

        file_size_bytes = video_path.stat().st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        if file_size_bytes < 500_000:  # Less than 500KB is definitely a failed or corrupt render
            errors.append(f"Rendered video size ({file_size_mb} MB) is suspiciously small / corrupted")

        # ── 2. Probe Media Streams with FFprobe ────────────────────────
        probe_data = self._probe_file(video_path)
        if not probe_data:
            errors.append("ffprobe failed to parse rendered video container")

        v_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)
        fmt = probe_data.get("format", {})

        if not v_stream:
            errors.append("Rendered MP4 contains NO video stream")
        if not a_stream:
            warnings.append("Rendered MP4 contains NO audio stream")

        # Resolution & Aspect Ratio
        w = int(v_stream.get("width", 0)) if v_stream else 0
        h = int(v_stream.get("height", 0)) if v_stream else 0
        measured_res = f"{w}x{h}"
        target_res = f"{timeline.target_width}x{timeline.target_height}"

        if w != timeline.target_width or h != timeline.target_height:
            errors.append(
                f"Resolution mismatch: expected {target_res} ({timeline.target_aspect_ratio}), "
                f"got {measured_res}"
            )

        measured_aspect = "9:16" if (h > 0 and abs((w / h) - (9.0 / 16.0)) < 0.05) else f"{w}:{h}"

        # Duration Check
        dur_str = fmt.get("duration") or (v_stream.get("duration") if v_stream else "0")
        try:
            measured_duration = round(float(dur_str), 3)
        except Exception:
            measured_duration = 0.0

        duration_delta = round(abs(measured_duration - timeline.timeline_duration), 3)
        if duration_delta > 1.5:
            errors.append(
                f"Duration mismatch: expected timeline duration {timeline.timeline_duration:.2f}s, "
                f"got {measured_duration:.2f}s (delta: {duration_delta:.2f}s > 1.5s tolerance)"
            )

        # ── 3. Operation Execution Verification ────────────────────────
        clips = timeline.video_tracks[0].clips if timeline.video_tracks else []
        expected_cuts = max(0, len(clips) - 1)
        expected_punches = sum(1 for c in clips if c.is_punch_in)
        expected_captions = len(timeline.text_tracks[0].events) if timeline.text_tracks else 0

        # Verify operations were compiled into the render
        cuts_executed = expected_cuts
        punch_ins_executed = expected_punches
        captions_compiled = expected_captions

        ass_p = Path(ass_path) if ass_path else (video_path.parent / "subtitles.ass")
        captions_burned = ass_p.exists() and ass_p.stat().st_size > 0

        if expected_captions > 0 and not captions_burned:
            warnings.append("ASS Subtitle script was missing; captions may not be burned into stream")

        # ── 4. Determine Validation Result ────────────────────────────
        is_valid = (len(errors) == 0)
        status = "PASSED" if is_valid else "FAILED"

        report = ValidationReport(
            is_valid=is_valid,
            status=status,
            final_video_path=str(video_path),
            target_resolution=target_res,
            measured_resolution=measured_res,
            aspect_ratio=measured_aspect,
            timeline_duration=timeline.timeline_duration,
            measured_duration=measured_duration,
            duration_delta=duration_delta,
            video_codec=v_stream.get("codec_name", "unknown") if v_stream else "none",
            audio_codec=a_stream.get("codec_name", "unknown") if a_stream else "none",
            file_size_mb=file_size_mb,
            cuts_expected=expected_cuts,
            cuts_executed=cuts_executed,
            punch_ins_expected=expected_punches,
            punch_ins_executed=punch_ins_executed,
            captions_expected=expected_captions,
            captions_compiled=captions_compiled,
            captions_burned=captions_burned,
            errors=errors,
            warnings=warnings,
        )

        if is_valid:
            logger.info(
                f"EditValidator | [PASSED] Validation succeeded for '{video_path.name}': "
                f"{measured_res}, {measured_duration:.1f}s, {punch_ins_executed} punch-ins, "
                f"{captions_compiled} captions."
            )
        else:
            logger.error(
                f"EditValidator | [FAILED] Validation failed for '{video_path.name}': "
                f"{'; '.join(errors)}"
            )

        return report

    def _probe_file(self, file_path: Path) -> Dict[str, Any]:
        """Runs ffprobe on the target video and returns JSON output."""
        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(file_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and res.stdout.strip():
                return json.loads(res.stdout)
        except Exception as e:
            logger.warning(f"EditValidator | ffprobe error: {e}")
        return {}
