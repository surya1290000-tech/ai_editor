"""
core/timeline/reframe.py

Face-Aware 9:16 Auto-Reframe Engine
────────────────────────────────────
Computes optimal 9:16 vertical crop windows (1080×1920) preserving the speaker's
face position and headroom using MediaPipe facial tracking telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from core.timeline.models import CropRect


@dataclass
class ReframePlan:
    target_width: int = 1080
    target_height: int = 1920
    target_aspect_ratio: str = "9:16"
    base_crop: CropRect = None
    punch_in_crop: CropRect = None
    face_center_x: float = 0.5
    face_center_y: float = 0.4
    ffmpeg_base_filter: str = ""
    ffmpeg_punch_filter: str = ""


def auto_reframe(
    source_width: int,
    source_height: int,
    face_tracking_data: Optional[Dict[str, Any]] = None,
    target_aspect_ratio: float = 9.0 / 16.0,
    punch_in_scale: float = 1.25,
) -> ReframePlan:
    """
    Computes face-aware vertical 9:16 reframe coordinates based on MediaPipe tracking.
    """
    face_tracking_data = face_tracking_data or {}
    per_frame = face_tracking_data.get("per_frame", [])

    # Extract all detected face centers
    detected_xs: List[float] = []
    detected_ys: List[float] = []

    for f in per_frame:
        if f.get("face_detected"):
            cx = f.get("face_center_x")
            cy = f.get("face_center_y")
            if cx is not None and 0.0 <= cx <= 1.0:
                detected_xs.append(cx)
            if cy is not None and 0.0 <= cy <= 1.0:
                detected_ys.append(cy)

    # Compute stable face anchor (median to resist outliers)
    if detected_xs:
        sorted_xs = sorted(detected_xs)
        face_cx = sorted_xs[len(sorted_xs) // 2]
    else:
        face_cx = 0.5

    if detected_ys:
        sorted_ys = sorted(detected_ys)
        face_cy = sorted_ys[len(sorted_ys) // 2]
    else:
        face_cy = 0.42

    # ── 1. Compute Base 9:16 Crop (1.0x) ──────────────────────────
    src_aspect = source_width / max(1.0, float(source_height))
    if src_aspect >= target_aspect_ratio:
        # Wider than 9:16 (horizontal, square, etc.): height is limiting dimension
        crop_h = source_height
        crop_w = int(round(crop_h * target_aspect_ratio))
        crop_w = max(2, min(source_width, (crop_w // 2) * 2))
        center_pixel_x = int(round(face_cx * source_width))
        x_offset = center_pixel_x - (crop_w // 2)
        x_offset = max(0, min(source_width - crop_w, x_offset))
        y_offset = 0
    else:
        # Narrower than 9:16: width is limiting dimension
        crop_w = source_width
        crop_h = int(round(crop_w / target_aspect_ratio))
        crop_h = max(2, min(source_height, (crop_h // 2) * 2))
        x_offset = 0
        center_pixel_y = int(round(face_cy * source_height))
        y_offset = center_pixel_y - (crop_h // 2)
        y_offset = max(0, min(source_height - crop_h, y_offset))

    base_crop = CropRect(
        x=round(x_offset / source_width, 4),
        y=round(y_offset / source_height, 4),
        w=round(crop_w / source_width, 4),
        h=round(crop_h / source_height, 4),
    )

    base_filter = f"crop=w={crop_w}:h={crop_h}:x={x_offset}:y={y_offset},scale=1080:1920"

    # ── 2. Compute Punch-In Crop (1.25x Zoom) ─────────────────────
    punch_w = int(round(crop_w / punch_in_scale))
    punch_h = int(round(crop_h / punch_in_scale))
    punch_w = max(2, min(source_width, (punch_w // 2) * 2))
    punch_h = max(2, min(source_height, (punch_h // 2) * 2))

    # Center punch-in horizontally on face, vertically weighted toward upper third
    center_pixel_x = int(round(face_cx * source_width))
    punch_x = center_pixel_x - (punch_w // 2)
    punch_x = max(0, min(source_width - punch_w, punch_x))

    center_pixel_y = int(round(face_cy * source_height))
    punch_y = center_pixel_y - int(round(punch_h * 0.38))
    punch_y = max(0, min(source_height - punch_h, punch_y))

    punch_crop = CropRect(
        x=round(punch_x / source_width, 4),
        y=round(punch_y / source_height, 4),
        w=round(punch_w / source_width, 4),
        h=round(punch_h / source_height, 4),
    )

    punch_filter = f"crop=w={punch_w}:h={punch_h}:x={punch_x}:y={punch_y},scale=1080:1920"

    return ReframePlan(
        target_width=1080,
        target_height=1920,
        target_aspect_ratio="9:16",
        base_crop=base_crop,
        punch_in_crop=punch_crop,
        face_center_x=round(face_cx, 3),
        face_center_y=round(face_cy, 3),
        ffmpeg_base_filter=base_filter,
        ffmpeg_punch_filter=punch_filter,
    )
