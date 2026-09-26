"""
agents/agent1_analyzer/stages/s1_media_extraction.py

Stage 1: Media Extraction
─────────────────────────
INPUT:  video file path
OUTPUT: audio.wav (16kHz mono), frames/ directory (1 jpg/second),
        media_meta.json (duration, fps, resolution, scene info)

Tools: FFmpeg (via ffmpeg-python), PySceneDetect

Design principles:
- All extraction is deterministic and lossless of information
- Audio extracted at 16kHz mono (Whisper requirement)
- Frames sampled at configurable fps (default 1fps for CPU-friendliness)
- Scene detection is a fast quality check, not a primary analysis tool
- Every operation is wrapped for clear error reporting
- Output paths are deterministic and recorded in media_meta.json
"""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from core.logger import get_logger

logger = get_logger(__name__)


class MediaExtractionError(Exception):
    """Raised when a critical media extraction step fails."""


# ─── Public entry point ───────────────────────────────────────────────────────

def run(
    video_path: Path,
    output_dir: Path,
    frame_sample_fps: int = 1,
) -> dict:
    """
    Run Stage 1: Extract audio, frames, and metadata from a video file.

    Args:
        video_path:      Path to the source video file.
        output_dir:      Directory where all stage outputs are written.
        frame_sample_fps: How many frames per second to extract (default 1).

    Returns:
        media_meta dict (also saved as output_dir/media_meta.json)

    Raises:
        MediaExtractionError on any critical failure.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Stage 1 | Media extraction starting: {video_path.name}")

    # ── 1. Validate input ─────────────────────────────────────────
    _validate_input(video_path)

    # ── 2. Probe metadata ─────────────────────────────────────────
    logger.info("Stage 1 | Probing video metadata...")
    meta = _probe_metadata(video_path)
    logger.info(
        f"Stage 1 | Duration={meta['duration_seconds']:.1f}s  "
        f"Resolution={meta['width']}x{meta['height']}  "
        f"FPS={meta['fps']:.2f}  "
        f"Has_audio={meta['has_audio']}"
    )

    # ── 3. Validate format ────────────────────────────────────────
    warnings = _validate_format(meta)
    for w in warnings:
        logger.warning(f"Stage 1 | {w}")

    # ── 4. Extract audio ──────────────────────────────────────────
    audio_path = output_dir / "audio.wav"
    logger.info("Stage 1 | Extracting audio (16kHz mono WAV)...")
    _extract_audio(video_path, audio_path)
    logger.info(f"Stage 1 | Audio saved: {audio_path.name} ({audio_path.stat().st_size // 1024} KB)")

    # ── 5. Extract frames ─────────────────────────────────────────
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    logger.info(f"Stage 1 | Extracting frames at {frame_sample_fps} fps...")
    frame_count = _extract_frames(video_path, frames_dir, frame_sample_fps)
    logger.info(f"Stage 1 | {frame_count} frames extracted")

    # ── 6. Scene detection ────────────────────────────────────────
    logger.info("Stage 1 | Running scene detection...")
    scene_info = _detect_scenes(video_path, meta["duration_seconds"])
    if scene_info["scene_count"] > 1:
        logger.warning(
            f"Stage 1 | {scene_info['scene_count']} scenes detected in source footage. "
            "This may be a pre-cut clip, not a raw single take."
        )
    else:
        logger.info("Stage 1 | Single scene detected (expected for raw Reel footage)")

    # ── 7. Assemble and save media_meta ───────────────────────────
    media_meta = {
        "source_file": video_path.name,
        "source_path_absolute": str(video_path.resolve()),
        **meta,
        "audio_path": str(audio_path),
        "frames_dir": str(frames_dir),
        "frame_count_extracted": frame_count,
        "frame_sample_fps": frame_sample_fps,
        "scene_info": scene_info,
        "quality_warnings": warnings,
    }

    meta_path = output_dir / "media_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(media_meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Stage 1 | Complete. meta saved: {meta_path.name}")
    return media_meta


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _validate_input(video_path: Path) -> None:
    """Ensure the input file exists and is a recognised video format."""
    if not video_path.exists():
        raise MediaExtractionError(f"Video file not found: {video_path}")

    allowed_extensions = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
    if video_path.suffix.lower() not in allowed_extensions:
        raise MediaExtractionError(
            f"Unsupported file format '{video_path.suffix}'. "
            f"Supported: {', '.join(sorted(allowed_extensions))}"
        )

    if not shutil.which("ffmpeg"):
        raise MediaExtractionError(
            "FFmpeg not found on PATH. "
            "Install FFmpeg and ensure it is available: https://www.gyan.dev/ffmpeg/builds/"
        )


def _probe_metadata(video_path: Path) -> dict:
    """
    Use ffprobe to extract video/audio stream metadata.
    Returns a clean dict of the most useful properties.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise MediaExtractionError(
            f"ffprobe failed on '{video_path.name}': {result.stderr.strip()}"
        )

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    # Extract video stream
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), {}
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), {}
    )

    # Parse FPS from avg_frame_rate (e.g. "30000/1001")
    fps = 0.0
    fps_str = video_stream.get("avg_frame_rate", "0/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    else:
        fps = float(fps_str) if fps_str else 0.0

    duration = float(
        video_stream.get("duration")
        or fmt.get("duration")
        or 0.0
    )

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # Aspect ratio
    aspect_ratio = "unknown"
    if width > 0 and height > 0:
        if abs((width / height) - (9 / 16)) < 0.05:
            aspect_ratio = "9:16"
        elif abs((width / height) - (16 / 9)) < 0.05:
            aspect_ratio = "16:9"
        elif abs((width / height) - 1.0) < 0.05:
            aspect_ratio = "1:1"
        else:
            aspect_ratio = f"{width}:{height}"

    return {
        "duration_seconds": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "aspect_ratio": aspect_ratio,
        "video_codec": video_stream.get("codec_name", "unknown"),
        "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else None,
        "audio_sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        "audio_channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
        "has_audio": bool(audio_stream),
        "file_size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 1),
        "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000, 0),
    }


def _validate_format(meta: dict) -> list[str]:
    """
    Check for potential issues in the video format.
    Returns a list of warning strings (empty = no issues).
    """
    warnings = []

    if meta["aspect_ratio"] != "9:16":
        warnings.append(
            f"Aspect ratio is {meta['aspect_ratio']}, not 9:16. "
            "Instagram Reels are vertical (9:16). "
            "Analysis will continue but framing analysis may be less accurate."
        )

    dur = meta["duration_seconds"]
    if dur < 5:
        warnings.append(f"Video is very short ({dur:.1f}s). Analysis may have limited value.")
    elif dur > 180:
        warnings.append(
            f"Video is {dur:.1f}s long (>{180}s). "
            "Processing will take significantly longer. Consider trimming to the relevant section."
        )

    if not meta["has_audio"]:
        warnings.append("No audio stream detected. Speech analysis will be skipped.")

    if meta["fps"] > 0 and meta["fps"] < 23:
        warnings.append(f"Low frame rate ({meta['fps']:.1f} fps). Visual analysis may be less accurate.")

    return warnings


def _extract_audio(video_path: Path, output_path: Path) -> None:
    """
    Extract audio as 16kHz mono WAV.
    16kHz mono is the exact format Whisper expects — no resampling needed during transcription.
    """
    cmd = [
        "ffmpeg",
        "-y",                      # overwrite output
        "-i", str(video_path),
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", "16000",            # 16kHz sample rate
        "-ac", "1",                # mono
        "-loglevel", "error",      # suppress progress output
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise MediaExtractionError(
            f"Audio extraction failed: {result.stderr.strip()}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise MediaExtractionError("Audio extraction produced an empty file.")


def _extract_frames(
    video_path: Path,
    frames_dir: Path,
    fps: int,
) -> int:
    """
    Extract frames at the specified rate as JPEG images.
    Frames are named frame_0001.jpg, frame_0002.jpg, etc.
    Returns the number of frames extracted.
    """
    output_pattern = str(frames_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",       # sample at target fps
        "-q:v", "3",               # JPEG quality (2=best, 5=default, good for analysis)
        "-loglevel", "error",
        output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise MediaExtractionError(
            f"Frame extraction failed: {result.stderr.strip()}"
        )

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    return len(frames)


def _detect_scenes(video_path: Path, duration_seconds: float) -> dict:
    """
    Run PySceneDetect for a fast content-aware scene detection pass.

    For raw single-take Reels, we expect exactly 1 scene.
    Multiple scenes = the footage was pre-cut, or there was a camera interruption.

    Falls back gracefully if scenedetect is not installed.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        # Threshold 27 = standard content-detect sensitivity
        # min_scene_len=0.5s prevents very short false-positive scenes
        scene_manager.add_detector(ContentDetector(threshold=27, min_scene_len=15))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for i, (start, end) in enumerate(scene_list):
            scenes.append({
                "scene_id": i,
                "start_seconds": round(start.get_seconds(), 3),
                "end_seconds": round(end.get_seconds(), 3),
                "duration_seconds": round(end.get_seconds() - start.get_seconds(), 3),
            })

        return {
            "scene_count": len(scenes),
            "scenes": scenes,
            "is_single_take": len(scenes) <= 1,
            "detection_method": "ContentDetector(threshold=27)",
        }

    except ImportError:
        logger.warning("Stage 1 | scenedetect not installed — scene detection skipped.")
        return {
            "scene_count": None,
            "scenes": [],
            "is_single_take": None,
            "detection_method": "skipped (scenedetect not installed)",
        }
    except Exception as e:
        logger.warning(f"Stage 1 | Scene detection failed: {e}")
        return {
            "scene_count": None,
            "scenes": [],
            "is_single_take": None,
            "detection_method": f"failed: {str(e)}",
        }
