"""
core/timeline/composer.py

Multi-Track Timeline Composer
────────────────────────────
Compiles Agent 2's creative decisions into the authoritative Timeline IR.
Deterministic intermediate representation linking every edit operation back
to its analytical origin opportunity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

from core.timeline.models import (
    TimelineIR, VideoTrack, VideoClip, Transition, Transform2D, CropRect
)
from core.timeline.remapper import TimecodeRemapper
from core.timeline.reframe import auto_reframe, ReframePlan
from core.timeline.captions import CaptionCompiler
from core.timeline.broll import BrollAssetResolver
from core.timeline.audio import AudioTimelineBuilder
from core.logger import get_logger

logger = get_logger(__name__)


class TimelineComposer:
    """
    Composes creative decisions and multi-modal observations into an executable Timeline IR.
    """

    def compose(
        self,
        source_video_path: Path | str,
        media_meta: Dict[str, Any],
        creative_decisions: Dict[str, Any],
        transcript_raw: Dict[str, Any],
        fused_timeline: Dict[str, Any],
        visual_features: Optional[Dict[str, Any]] = None,
    ) -> tuple[TimelineIR, ReframePlan]:
        """
        Builds the complete Timeline IR.
        """
        source_video_str = str(Path(source_video_path).resolve())
        src_w = int(media_meta.get("width", 2160))
        src_h = int(media_meta.get("height", 1192))
        src_dur = float(media_meta.get("duration_seconds", 90.833))

        keep_segments = creative_decisions.get("keep_segments", [])
        punch_ins = creative_decisions.get("punch_ins", [])
        style = creative_decisions.get("creative_style", {})
        framing_mode = style.get("framing_mode", "9:16_vertical")
        punch_scale = float(style.get("punch_in_scale", 1.25))

        # ── 1. Authoritative Timecode Remapper ─────────────────────────
        remapper = TimecodeRemapper(keep_segments)
        logger.info(
            f"Composer | Remapper initialized with {len(keep_segments)} keep segments. "
            f"Master duration: {remapper.total_timeline_duration:.2f}s (trimmed {len(remapper.cuts)} cuts)"
        )

        # ── 2. Face-Aware 9:16 Auto-Reframe ───────────────────────────
        is_vertical = (framing_mode == "9:16_vertical")
        reframe_plan = auto_reframe(
            source_width=src_w,
            source_height=src_h,
            face_tracking_data=visual_features,
            target_aspect_ratio=9.0 / 16.0 if is_vertical else float(src_w) / float(src_h),
            punch_in_scale=punch_scale,
        )

        # ── 3. Build Video Track V1 with Explicit Jump-Cut Punch-Ins ──
        video_clips: List[VideoClip] = []
        transitions: List[Transition] = []
        clip_counter = 1

        timeline_cursor = 0.0

        for seg in keep_segments:
            s_start = float(seg["start"])
            s_end = float(seg["end"])

            # Find all punch-in windows falling inside this keep segment
            seg_punches: List[Dict[str, Any]] = []
            for p in punch_ins:
                p_start = float(p["source_start"])
                p_end = float(p["source_end"])

                # Check overlap with current segment
                overlap_start = max(s_start, p_start)
                overlap_end = min(s_end, p_end)
                if overlap_end - overlap_start >= 0.4:
                    seg_punches.append({
                        "start": overlap_start,
                        "end": overlap_end,
                        "scale": float(p.get("scale", punch_scale)),
                        "reason": p.get("reason", "Punch-in on thesis peak"),
                        "confidence": float(p.get("confidence", 0.9)),
                        "origin_opportunity_id": p.get("origin_opportunity_id"),
                    })

            if not seg_punches:
                # Continuous base shot for this segment
                dur = round(s_end - s_start, 4)
                t_in = timeline_cursor
                t_out = round(timeline_cursor + dur, 4)
                timeline_cursor = t_out

                video_clips.append(VideoClip(
                    clip_id=f"vclip_{clip_counter:03d}",
                    source_reference=source_video_str,
                    source_in=s_start,
                    source_out=s_end,
                    timeline_in=t_in,
                    timeline_out=t_out,
                    crop=reframe_plan.base_crop,
                    scale=1.0,
                    is_punch_in=False,
                    reason="Base master framing",
                    confidence=1.0,
                ))
                clip_counter += 1
                continue

            # Sub-slice segment: Base -> Punch-In Jump Cut -> Base
            seg_punches.sort(key=lambda x: x["start"])
            curr = s_start

            for punch in seg_punches:
                p_in = punch["start"]
                p_out = punch["end"]

                # Pre-punch base shot
                if p_in - curr >= 0.3:
                    dur = round(p_in - curr, 4)
                    t_in = timeline_cursor
                    t_out = round(timeline_cursor + dur, 4)
                    timeline_cursor = t_out

                    video_clips.append(VideoClip(
                        clip_id=f"vclip_{clip_counter:03d}",
                        source_reference=source_video_str,
                        source_in=round(curr, 4),
                        source_out=round(p_in, 4),
                        timeline_in=t_in,
                        timeline_out=t_out,
                        crop=reframe_plan.base_crop,
                        scale=1.0,
                        is_punch_in=False,
                        reason="Base shot before emphasis punch",
                        confidence=1.0,
                    ))
                    clip_counter += 1

                # The Punch-In Jump Cut
                p_dur = round(p_out - p_in, 4)
                t_in = timeline_cursor
                t_out = round(timeline_cursor + p_dur, 4)
                timeline_cursor = t_out

                video_clips.append(VideoClip(
                    clip_id=f"vclip_{clip_counter:03d}",
                    source_reference=source_video_str,
                    source_in=round(p_in, 4),
                    source_out=round(p_out, 4),
                    timeline_in=t_in,
                    timeline_out=t_out,
                    crop=reframe_plan.punch_in_crop,
                    scale=punch["scale"],
                    is_punch_in=True,
                    origin_opportunity_id=punch["origin_opportunity_id"],
                    reason=punch["reason"],
                    confidence=punch["confidence"],
                ))
                # Add jump-cut transition marker
                transitions.append(Transition(
                    transition_id=f"trans_{clip_counter:03d}",
                    transition_type="HARD_CUT",
                    timeline_time=t_in,
                    duration=0.0,
                    incoming_clip_id=f"vclip_{clip_counter:03d}",
                    origin_opportunity_id=punch["origin_opportunity_id"],
                ))
                clip_counter += 1
                curr = p_out

            # Post-punch tail base shot
            if s_end - curr >= 0.15:
                dur = round(s_end - curr, 4)
                t_in = timeline_cursor
                t_out = round(timeline_cursor + dur, 4)
                timeline_cursor = t_out

                video_clips.append(VideoClip(
                    clip_id=f"vclip_{clip_counter:03d}",
                    source_reference=source_video_str,
                    source_in=round(curr, 4),
                    source_out=round(s_end, 4),
                    timeline_in=t_in,
                    timeline_out=t_out,
                    crop=reframe_plan.base_crop,
                    scale=1.0,
                    is_punch_in=False,
                    reason="Return to base framing after emphasis punch",
                    confidence=1.0,
                ))
                clip_counter += 1

        main_video_track = VideoTrack(track_id="track_video_v1", name="Primary Speaker Track", clips=video_clips)

        # ── 4. Build Text Track (Captions) ────────────────────────────
        caption_compiler = CaptionCompiler(
            remapper=remapper,
            target_width=reframe_plan.target_width,
            target_height=reframe_plan.target_height,
            font_size=72 if is_vertical else 52,
            margin_v=380 if is_vertical else 130,
        )
        caption_track = caption_compiler.compile(
            transcript_raw=transcript_raw,
            fused_timeline=fused_timeline,
            emphasis_directives=creative_decisions.get("caption_directives", {}).get("emphasis_tokens", []),
        )

        # ── 5. Build B-Roll Overlay Track ─────────────────────────────
        broll_resolver = BrollAssetResolver()
        broll_track, unresolved_broll = broll_resolver.build_broll_track(
            broll_requests=creative_decisions.get("broll_requests", []),
            remapper=remapper,
        )

        # ── 6. Build Audio Tracks (Voice + SFX) ───────────────────────
        audio_builder = AudioTimelineBuilder()
        audio_tracks, unresolved_sfx = audio_builder.build_audio_tracks(
            source_audio_path=source_video_str,
            keep_segments=keep_segments,
            sound_requests=creative_decisions.get("sound_requests", []),
            remapper=remapper,
        )

        # ── 7. Assemble Full Timeline IR ──────────────────────────────
        all_unresolved = unresolved_broll + unresolved_sfx

        timeline_ir = TimelineIR(
            version="2.0.0",
            project_name=f"Reel_{Path(source_video_str).stem}",
            source_video=source_video_str,
            source_duration=src_dur,
            timeline_duration=remapper.total_timeline_duration,
            target_width=reframe_plan.target_width,
            target_height=reframe_plan.target_height,
            target_aspect_ratio="9:16" if is_vertical else "16:9",
            fps=float(media_meta.get("fps", 30.0)),
            video_tracks=[main_video_track],
            broll_tracks=[broll_track],
            text_tracks=[caption_track],
            audio_tracks=audio_tracks,
            transitions=transitions,
            unresolved_items=all_unresolved,
            metadata={
                "cuts_count": len(remapper.cuts),
                "subclips_count": len(video_clips),
                "punch_ins_count": sum(1 for c in video_clips if c.is_punch_in),
                "captions_count": len(caption_track.events),
                "framing_mode": framing_mode,
                "face_center": {"x": reframe_plan.face_center_x, "y": reframe_plan.face_center_y},
            },
        )

        return timeline_ir, reframe_plan
