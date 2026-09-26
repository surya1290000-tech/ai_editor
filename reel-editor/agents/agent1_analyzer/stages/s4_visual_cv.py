"""
agents/agent1_analyzer/stages/s4_visual_cv.py

Stage 4: Visual Signal Analysis
────────────────────────────────
INPUT:  frames/ directory (JPEG frames from Stage 1)
        media_meta.json  (fps, duration, etc.)
OUTPUT: visual_features.json

Tools: OpenCV, MediaPipe FaceDetection

What this stage measures (all deterministic — no LLM):
  - Per-frame: face detection, bounding box, face size, face position
  - Framing classification per frame: ECU / CU / MCU / MS / WS / no_face
  - Headroom measurement
  - Camera motion magnitude (optical flow between consecutive frames)
  - Camera shake score (optical flow vector variance)
  - Per-frame brightness and sharpness
  - Shot segment grouping (consecutive frames with consistent framing)
  - Punch-in candidacy per segment (purely technical — needs semantic context to become a recommendation)
  - Video quality warnings

What this stage does NOT do:
  - Story understanding
  - Creative decisions
  - B-roll recommendations
  - Any LLM calls

Design notes:
  - Frames are already at 1fps from Stage 1, so analysis loops over JPEGs (fast)
  - MediaPipe FaceDetection model_selection=0 is optimised for faces within 2m (selfie/talking-head)
  - Optical flow is computed between consecutive frames for camera motion
  - Framing is classified purely from face_area_ratio (face bbox area / frame area)
  - Shot segments group consecutive frames with the same dominant framing class
  - All classifications are rule-based, fully auditable
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)

# ─── Framing thresholds (face_area_ratio = face_w * face_h relative to frame) ─
_FRAMING_THRESHOLDS = {
    "ECU": 0.35,   # face covers >35% of frame area
    "CU":  0.20,   # 20–35%
    "MCU": 0.08,   # 8–20% ← most common for Reels
    "MS":  0.03,   # 3–8%
    "WS":  0.0,    # <3%
}

# ─── Camera motion classification ─────────────────────────────────────────────
_MOTION_THRESHOLDS = {
    "static":       1.5,   # px/frame mean optical flow
    "subtle_drift": 3.0,
    "movement":     8.0,
    # above 8.0 = "shake" (also checked via variance)
}
_SHAKE_VARIANCE_THRESHOLD = 15.0   # high variance = shaky vectors


# ─── Public entry point ───────────────────────────────────────────────────────

def run(
    frames_dir: Path,
    media_meta: dict,
    output_dir: Path,
    mediapipe_complexity: int = 0,
) -> dict:
    """
    Run Stage 4: Analyze frames for face position, framing, and camera motion.

    Args:
        frames_dir:           Path to frames/ directory (JPEG files).
        media_meta:           Dict from Stage 1 (media_meta.json).
        output_dir:           Directory to write visual_features.json.
        mediapipe_complexity: 0=fast (CPU-optimised), 1=balanced, 2=best.

    Returns:
        visual_features dict (also saved as output_dir/visual_features.json)
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    frame_sample_fps = media_meta.get("frame_sample_fps", 1)

    # Sort frames deterministically
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    logger.info(f"Stage 4 | Analyzing {len(frame_files)} frames ({frame_sample_fps} fps sample)...")

    # Initialise MediaPipe face detector
    detector = _init_face_detector(mediapipe_complexity)

    # ── Per-frame analysis ────────────────────────────────────────
    per_frame = []
    prev_gray = None
    frame_w = media_meta.get("width", 1)
    frame_h = media_meta.get("height", 1)

    for i, frame_file in enumerate(frame_files):
        timestamp = i / frame_sample_fps   # seconds

        frame_bgr = cv2.imread(str(frame_file))
        if frame_bgr is None:
            logger.warning(f"Stage 4 | Could not read frame: {frame_file.name}")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        # Face detection
        face_data = _detect_face(detector, frame_rgb, w, h)

        # Framing classification
        framing = _classify_framing(face_data.get("face_area_ratio", 0.0), face_data["face_detected"])

        # Headroom
        headroom = _compute_headroom(face_data, h)

        # Camera motion (optical flow vs previous frame)
        motion_data = _compute_motion(frame_bgr, prev_gray)
        prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Visual quality
        brightness = _compute_brightness(frame_bgr)
        sharpness = _compute_sharpness(frame_bgr)

        per_frame.append({
            "frame_index": i,
            "timestamp": round(timestamp, 3),
            "filename": frame_file.name,
            **face_data,
            "framing": framing,
            "headroom": headroom,
            **motion_data,
            "brightness": brightness,
            "sharpness": sharpness,
        })

    logger.info(f"Stage 4 | {len(per_frame)} frames analyzed")

    # ── Shot segments ─────────────────────────────────────────────
    shot_segments = _build_shot_segments(per_frame, frame_sample_fps)
    logger.info(f"Stage 4 | {len(shot_segments)} shot segments identified")

    # ── Framing summary ───────────────────────────────────────────
    framing_counts = {}
    for f in per_frame:
        framing_counts[f["framing"]] = framing_counts.get(f["framing"], 0) + 1
    dominant_framing = max(framing_counts, key=framing_counts.get) if framing_counts else "unknown"
    face_detected_ratio = round(
        sum(1 for f in per_frame if f["face_detected"]) / max(len(per_frame), 1), 3
    )

    # ── Quality warnings ──────────────────────────────────────────
    quality_warnings = _generate_quality_warnings(per_frame, face_detected_ratio, media_meta)
    for w in quality_warnings:
        logger.warning(f"Stage 4 | {w['description']}")

    visual_features = {
        "frames_analyzed": len(per_frame),
        "frame_sample_fps": frame_sample_fps,
        "dominant_framing": dominant_framing,
        "framing_distribution": framing_counts,
        "face_detected_ratio": face_detected_ratio,
        "shot_segments": shot_segments,
        "per_frame": per_frame,
        "quality_warnings": quality_warnings,
    }

    from core.json_utils import save_json
    out_path = output_dir / "visual_features.json"
    save_json(visual_features, out_path)

    logger.info(f"Stage 4 | Complete. Visual features saved: {out_path.name}")
    logger.info(
        f"Stage 4 | Summary: dominant_framing={dominant_framing}  "
        f"face_detected={face_detected_ratio:.0%}  "
        f"segments={len(shot_segments)}"
    )
    return visual_features


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _init_face_detector(complexity: int):
    """Initialise MediaPipe Tasks FaceDetector."""
    import urllib.request
    from pathlib import Path
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "blaze_face_short_range.tflite"

    if not model_path.exists() or model_path.stat().st_size == 0:
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, model_path)

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=0.5,
    )
    return FaceDetector.create_from_options(options)


def _detect_face(detector, frame_rgb: np.ndarray, frame_w: int, frame_h: int) -> dict:
    """
    Run MediaPipe face detection on a single frame.
    Returns normalised bounding box values (0.0–1.0 relative to frame size).
    """
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)

    if not results.detections:
        return {
            "face_detected": False,
            "face_bbox": None,
            "face_center_x": None,
            "face_center_y": None,
            "face_area_ratio": 0.0,
            "face_confidence": 0.0,
        }

    # Take the highest-confidence detection
    detection = max(results.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
    bb = detection.bounding_box

    xmin = round(max(0.0, bb.origin_x / frame_w), 3)
    ymin = round(max(0.0, bb.origin_y / frame_h), 3)
    w = round(min(1.0, bb.width / frame_w), 3)
    h = round(min(1.0, bb.height / frame_h), 3)

    cx = round(xmin + w / 2, 3)
    cy = round(ymin + h / 2, 3)
    area = round(w * h, 4)
    conf = round(float(detection.categories[0].score if detection.categories else 0.0), 3)

    return {
        "face_detected": True,
        "face_bbox": {
            "x": xmin,
            "y": ymin,
            "w": w,
            "h": h,
        },
        "face_center_x": cx,
        "face_center_y": cy,
        "face_area_ratio": area,
        "face_confidence": conf,
    }


def _classify_framing(area_ratio: float, face_detected: bool) -> str:
    """
    Classify the shot type based on face area relative to frame.
    Returns one of: ECU, CU, MCU, MS, WS, no_face
    """
    if not face_detected or area_ratio <= 0:
        return "no_face"
    if area_ratio >= _FRAMING_THRESHOLDS["ECU"]:
        return "ECU"
    elif area_ratio >= _FRAMING_THRESHOLDS["CU"]:
        return "CU"
    elif area_ratio >= _FRAMING_THRESHOLDS["MCU"]:
        return "MCU"
    elif area_ratio >= _FRAMING_THRESHOLDS["MS"]:
        return "MS"
    else:
        return "WS"


def _compute_headroom(face_data: dict, frame_h: int) -> Optional[str]:
    """
    Estimate headroom as the vertical space above the face (normalised).
    Returns: 'good' | 'tight' | 'excessive' | None
    """
    if not face_data["face_detected"]:
        return None
    bbox = face_data["face_bbox"]
    space_above = bbox["y"]   # normalised distance from top of frame to top of face
    if space_above < 0.04:
        return "tight"
    elif space_above > 0.22:
        return "excessive"
    return "good"


def _compute_motion(frame_bgr: np.ndarray, prev_gray: Optional[np.ndarray]) -> dict:
    """
    Compute optical flow between current and previous frame.
    Returns camera motion magnitude and classification.
    """
    if prev_gray is None:
        return {
            "motion_magnitude": 0.0,
            "motion_variance": 0.0,
            "camera_motion": "static",
        }

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Downscale for fast CPU optical flow (100x faster, identical camera motion signals)
    flow_w = 320
    flow_h = max(1, int(320 * h / w))
    prev_small = cv2.resize(prev_gray, (flow_w, flow_h))
    curr_small = cv2.resize(gray, (flow_w, flow_h))

    # Farneback optical flow — fast, no feature matching needed
    flow = cv2.calcOpticalFlowFarneback(
        prev_small, curr_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Magnitude of flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(magnitude))
    variance = float(np.var(magnitude))

    # Classify
    if variance > _SHAKE_VARIANCE_THRESHOLD:
        motion_class = "shake"
    elif mean_mag > _MOTION_THRESHOLDS["movement"]:
        motion_class = "movement"
    elif mean_mag > _MOTION_THRESHOLDS["subtle_drift"]:
        motion_class = "subtle_drift"
    else:
        motion_class = "static"

    return {
        "motion_magnitude": round(mean_mag, 3),
        "motion_variance": round(variance, 3),
        "camera_motion": motion_class,
    }


def _compute_brightness(frame_bgr: np.ndarray) -> float:
    """Mean luminance (0–255). Used for exposure quality check."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return round(float(np.mean(gray)), 1)


def _compute_sharpness(frame_bgr: np.ndarray) -> float:
    """
    Laplacian variance as focus/sharpness metric.
    Higher = sharper. Values below ~50 indicate soft/blurry footage.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)


def _build_shot_segments(per_frame: list, fps: int) -> list:
    """
    Group consecutive frames into shot segments based on framing consistency.
    A new segment starts when:
      - Framing class changes
      - Camera motion changes from stable to shake (or vice versa)
    """
    if not per_frame:
        return []

    segments = []
    seg_start = 0

    def _flush_segment(start_idx: int, end_idx: int) -> dict:
        frames = per_frame[start_idx:end_idx + 1]
        framings = [f["framing"] for f in frames]
        motions = [f["camera_motion"] for f in frames]
        brightnesses = [f["brightness"] for f in frames]
        sharpnesses = [f["sharpness"] for f in frames]
        face_ratios = [f["face_area_ratio"] for f in frames if f["face_detected"]]

        # Dominant framing (most common)
        framing_counts = {}
        for fr in framings:
            framing_counts[fr] = framing_counts.get(fr, 0) + 1
        dom_framing = max(framing_counts, key=framing_counts.get)

        # Stability: if any frame is "shake", mark shake
        if "shake" in motions:
            stability = "shake"
        elif "movement" in motions:
            stability = "movement"
        elif "subtle_drift" in motions:
            stability = "subtle_drift"
        else:
            stability = "static"

        face_ratio = round(sum(1 for f in frames if f["face_detected"]) / len(frames), 3)

        # Punch-in candidacy:
        # Technical requirement only — semantic need comes from Stage 6/7
        punch_in_candidate = (
            dom_framing in ("MCU", "MS", "WS") and  # room to crop closer
            stability in ("static", "subtle_drift") and  # stable enough to crop
            face_ratio >= 0.75   # face reliably visible
        )

        avg_brightness = round(float(np.mean(brightnesses)), 1)
        avg_sharpness = round(float(np.mean(sharpnesses)), 1)

        quality_issues = []
        if avg_brightness < 55:
            quality_issues.append(f"underexposed (brightness={avg_brightness:.0f})")
        elif avg_brightness > 205:
            quality_issues.append(f"overexposed (brightness={avg_brightness:.0f})")
        if avg_sharpness < 50:
            quality_issues.append(f"soft/blurry (sharpness={avg_sharpness:.0f})")
        if stability == "shake":
            quality_issues.append("camera shake detected")

        return {
            "id": len(segments),
            "start": frames[0]["timestamp"],
            "end": frames[-1]["timestamp"],
            "duration": round(frames[-1]["timestamp"] - frames[0]["timestamp"] + 1/fps, 3),
            "frame_count": len(frames),
            "dominant_framing": dom_framing,
            "framing_distribution": framing_counts,
            "face_detected_ratio": face_ratio,
            "avg_stability": stability,
            "avg_brightness": avg_brightness,
            "avg_sharpness": avg_sharpness,
            "punch_in_candidate": punch_in_candidate,
            "quality_issues": quality_issues,
        }

    for i in range(1, len(per_frame)):
        prev = per_frame[i - 1]
        curr = per_frame[i]

        framing_changed = curr["framing"] != prev["framing"]
        stability_changed = (
            (curr["camera_motion"] == "shake") != (prev["camera_motion"] == "shake")
        )

        if framing_changed or stability_changed:
            segments.append(_flush_segment(seg_start, i - 1))
            seg_start = i

    segments.append(_flush_segment(seg_start, len(per_frame) - 1))
    return segments


def _generate_quality_warnings(per_frame: list, face_ratio: float, meta: dict) -> list:
    """Generate quality warnings for the full video visual analysis."""
    warnings = []

    if face_ratio < 0.5:
        warnings.append({
            "type": "video",
            "severity": "warning",
            "timestamp": None,
            "description": f"Face detected in only {face_ratio:.0%} of frames. "
                           "Check framing or lighting.",
            "recommendation": "Ensure face is clearly visible and centred throughout.",
        })

    # Check for many dark or overexposed frames
    dark_frames = sum(1 for f in per_frame if f["brightness"] < 55)
    bright_frames = sum(1 for f in per_frame if f["brightness"] > 205)
    if dark_frames > len(per_frame) * 0.2:
        warnings.append({
            "type": "video",
            "severity": "warning",
            "timestamp": None,
            "description": f"{dark_frames}/{len(per_frame)} frames appear underexposed.",
            "recommendation": "Improve lighting or adjust exposure before analysis.",
        })
    if bright_frames > len(per_frame) * 0.2:
        warnings.append({
            "type": "video",
            "severity": "warning",
            "timestamp": None,
            "description": f"{bright_frames}/{len(per_frame)} frames appear overexposed.",
            "recommendation": "Reduce exposure or avoid shooting in direct harsh light.",
        })

    # Aspect ratio check
    if meta.get("aspect_ratio") != "9:16":
        warnings.append({
            "type": "video",
            "severity": "info",
            "timestamp": None,
            "description": f"Video is {meta.get('aspect_ratio')} not 9:16 (Instagram Reels native).",
            "recommendation": "Consider cropping to 9:16 for Instagram-native output.",
        })

    return warnings
