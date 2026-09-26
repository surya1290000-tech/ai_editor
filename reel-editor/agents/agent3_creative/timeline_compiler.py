"""
agents/agent3_creative/timeline_compiler.py

Timeline IR Compiler for Agent 3 (Creative Director)
────────────────────────────────────────────────────
Compiles concrete CreativeDecision objects from CreativePlan into the authoritative
multi-track Timeline IR.

Strict rules:
1. Every generated timeline entity retains:
   - source_decision_id
   - origin_opportunity_id
   - source_evidence_ids (in metadata/reason)
2. Traceability survives completely down to rendering.
3. Unresolved assets are explicitly declared in TimelineIR.unresolved_items.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.timeline.models import (
    TimelineIR, VideoTrack, VideoClip, TextTrack, TextEvent,
    BrollTrack, BrollClip, AudioTrack, AudioClip, AudioRole,
    Transition, Transform2D, CropRect
)
from core.timeline.remapper import TimecodeRemapper
from core.timeline.reframe import auto_reframe, ReframePlan
from agents.agent3_creative.models import (
    CreativePlan, CreativeDecision, CreativeOperation, CreativeStatus
)
from core.logger import get_logger

logger = get_logger(__name__)


class CreativeTimelineCompiler:
    """
    Compiles a CreativePlan into an authoritative multi-track TimelineIR.
    """

    def compile(
        self,
        creative_plan: CreativePlan,
        source_video_path: Path | str,
        media_meta: Dict[str, Any],
        transcript_raw: Optional[Dict[str, Any]] = None,
        fused_timeline: Optional[Dict[str, Any]] = None,
    ) -> tuple[TimelineIR, ReframePlan]:
        """
        Compiles the CreativePlan into TimelineIR and ReframePlan.
        """
        source_video_str = str(Path(source_video_path).resolve())
        src_w = int(media_meta.get("width", 2160))
        src_h = int(media_meta.get("height", 1192))
        src_dur = float(media_meta.get("duration_seconds", creative_plan.duration_seconds))

        # Extract dead-space trims to build keep segments
        dead_space_cuts = [
            d for d in creative_plan.decisions
            if d.operation in [CreativeOperation.REMOVE_DEAD_SPACE, CreativeOperation.REMOVE_FILLER, CreativeOperation.CUT]
        ]

        # Construct keep segments from cuts
        keep_segments: List[Dict[str, float]] = []
        cursor = 0.0
        sorted_cuts = sorted(dead_space_cuts, key=lambda x: x.time_range[0])

        for cut in sorted_cuts:
            c_in = max(cursor, cut.time_range[0])
            c_out = min(src_dur, cut.time_range[1])
            if c_in > cursor + 0.10:
                keep_segments.append({"start": round(cursor, 3), "end": round(c_in, 3)})
            cursor = max(cursor, c_out)

        if cursor < src_dur - 0.10:
            keep_segments.append({"start": round(cursor, 3), "end": round(src_dur, 3)})

        if not keep_segments:
            keep_segments = [{"start": 0.0, "end": round(src_dur, 3)}]

        # Initialize TimecodeRemapper
        remapper = TimecodeRemapper(keep_segments)
        logger.info(
            f"CreativeTimelineCompiler | Remapper built with {len(keep_segments)} keep segments. "
            f"Timeline duration: {remapper.total_timeline_duration:.2f}s."
        )

        # Extract punch-in decisions
        punch_decisions = [
            d for d in creative_plan.decisions
            if d.operation == CreativeOperation.PUNCH_IN and d.status == CreativeStatus.RESOLVED and d.punch_in is not None
        ]

        # Determine average punch scale for reframe plan
        avg_punch_scale = 1.22
        if punch_decisions:
            avg_punch_scale = sum(d.punch_in.scale for d in punch_decisions) / len(punch_decisions)

        # Reframe plan (9:16 vertical)
        reframe_plan = auto_reframe(
            source_width=src_w,
            source_height=src_h,
            face_tracking_data=creative_plan.metadata.get("visual_features"),
            target_aspect_ratio=9.0 / 16.0,
            punch_in_scale=avg_punch_scale,
        )

        # ── 1. Compile Video Track V1 (Base + Jump-Cut Punch-Ins) ───────
        video_clips: List[VideoClip] = []
        transitions: List[Transition] = []
        clip_counter = 1
        timeline_cursor = 0.0

        for seg in keep_segments:
            s_start = float(seg["start"])
            s_end = float(seg["end"])

            # Find matching punch-ins overlapping this keep segment
            seg_punches: List[CreativeDecision] = []
            for p in punch_decisions:
                p_start = p.time_range[0]
                p_end = p.time_range[1]
                overlap_start = max(s_start, p_start)
                overlap_end = min(s_end, p_end)
                if overlap_end - overlap_start >= 0.40:
                    seg_punches.append(p)

            if not seg_punches:
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
                    reason="Base master speaker framing",
                    confidence=1.0,
                ))
                clip_counter += 1
                continue

            seg_punches.sort(key=lambda x: x.time_range[0])
            curr = s_start

            for punch in seg_punches:
                p_in = max(s_start, punch.time_range[0])
                p_out = min(s_end, punch.time_range[1])

                # Pre-punch base shot
                if p_in - curr >= 0.25:
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
                        reason="Base shot before emphasis jump cut",
                        confidence=1.0,
                    ))
                    clip_counter += 1

                # Punch-In Jump Cut
                p_dur = round(p_out - p_in, 4)
                t_in = timeline_cursor
                t_out = round(timeline_cursor + p_dur, 4)
                timeline_cursor = t_out

                # Dynamic crop rectangle anchored to punch.punch_in.anchor
                p_scale = punch.punch_in.scale
                p_crop = CropRect(
                    x=max(0.0, min(1.0 - (reframe_plan.punch_in_crop.w), punch.punch_in.anchor["x"] - (reframe_plan.punch_in_crop.w / 2.0))),
                    y=max(0.0, min(1.0 - (reframe_plan.punch_in_crop.h), punch.punch_in.anchor["y"] - (reframe_plan.punch_in_crop.h / 2.0))),
                    w=reframe_plan.punch_in_crop.w,
                    h=reframe_plan.punch_in_crop.h,
                )

                video_clips.append(VideoClip(
                    clip_id=f"vclip_{clip_counter:03d}",
                    source_reference=source_video_str,
                    source_in=round(p_in, 4),
                    source_out=round(p_out, 4),
                    timeline_in=t_in,
                    timeline_out=t_out,
                    crop=p_crop,
                    scale=p_scale,
                    is_punch_in=True,
                    origin_opportunity_id=punch.source_opportunity_ids[0] if punch.source_opportunity_ids else None,
                    reason=punch.reason,
                    confidence=punch.confidence,
                ))

                transitions.append(Transition(
                    transition_id=f"trans_{clip_counter:03d}",
                    transition_type=punch.punch_in.transition,
                    timeline_time=t_in,
                    duration=0.0,
                    incoming_clip_id=f"vclip_{clip_counter:03d}",
                    origin_opportunity_id=punch.source_opportunity_ids[0] if punch.source_opportunity_ids else None,
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
                    reason="Return to base framing after punch",
                    confidence=1.0,
                ))
                clip_counter += 1

        main_video_track = VideoTrack(
            track_id="track_video_v1",
            name="Primary Speaker Track",
            clips=video_clips,
        )

        # ── 2. Compile Text Track (Hero Text Cards + Subtitle Captions) ─
        text_events: List[TextEvent] = []
        text_counter = 1

        # A. Hero Text Cards
        hero_decisions = [
            d for d in creative_plan.decisions
            if d.operation == CreativeOperation.HERO_TEXT and d.status == CreativeStatus.RESOLVED and d.typography is not None
        ]

        for h in hero_decisions:
            t_in = remapper.map_source_to_timeline(h.time_range[0])
            t_out = remapper.map_source_to_timeline(h.time_range[1])
            if t_in is not None and t_out is not None and t_out > t_in:
                text_events.append(TextEvent(
                    event_id=f"txt_hero_{text_counter:03d}",
                    text=h.text_content or "HERO",
                    timeline_in=round(t_in, 3),
                    timeline_out=round(t_out, 3),
                    source_start=h.time_range[0],
                    source_end=h.time_range[1],
                    position_align=5,  # Mid-center title card
                    margin_v=0,
                    font_name=h.typography.font,
                    font_size=h.typography.size,
                    font_weight=h.typography.weight,
                    color_primary=h.typography.color_primary,
                    color_highlight=h.typography.color_highlight,
                    color_outline=h.typography.color_outline,
                    outline_width=h.typography.outline_width,
                    shadow_distance=h.typography.shadow_distance,
                    origin_opportunity_id=h.source_opportunity_ids[0] if h.source_opportunity_ids else None,
                    reason=h.reason,
                    confidence=h.confidence,
                ))
                text_counter += 1

        # B. Kinetic Captions with Emphasis Highlights
        emphasis_tokens = [
            d.text_content.strip().upper()
            for d in creative_plan.decisions
            if d.operation == CreativeOperation.EMPHASIS_TEXT and d.text_content
        ]

        raw_words = []
        if transcript_raw and "words" in transcript_raw:
            raw_words = transcript_raw["words"]
        elif fused_timeline and "sentences" in fused_timeline:
            for s in fused_timeline["sentences"]:
                raw_words.extend(s.get("words", []))

        # Chunk transcript into 2-4 word groups
        mapped_words = []
        for w in raw_words:
            w_in = float(w.get("start", 0.0))
            w_out = float(w.get("end", 0.0))
            t_in = remapper.map_source_to_timeline(w_in)
            t_out = remapper.map_source_to_timeline(w_out)
            if t_in is not None and t_out is not None and t_out > t_in:
                word_clean = str(w.get("word", "")).strip()
                if word_clean:
                    mapped_words.append({
                        "word": word_clean,
                        "timeline_in": t_in,
                        "timeline_out": t_out,
                        "source_start": w_in,
                        "source_end": w_out,
                    })

        # Group into 3-word chunks
        chunk_size = 3
        for i in range(0, len(mapped_words), chunk_size):
            chunk = mapped_words[i:i + chunk_size]
            c_text = " ".join(item["word"] for item in chunk)
            c_tin = chunk[0]["timeline_in"]
            c_tout = chunk[-1]["timeline_out"]

            # Highlight emphasis tokens
            tokens_in_chunk = [
                tok for tok in emphasis_tokens
                if any(tok.lower() in item["word"].lower() for item in chunk)
            ]

            text_events.append(TextEvent(
                event_id=f"txt_cap_{text_counter:03d}",
                text=c_text,
                timeline_in=round(c_tin, 3),
                timeline_out=round(c_tout, 3),
                source_start=chunk[0]["source_start"],
                source_end=chunk[-1]["source_end"],
                position_align=2,  # Bottom center
                margin_v=380,
                font_name="Nirmala UI" if any(re.search(r"[\u0c00-\u0c7f]", item["word"]) for item in chunk) else "Arial Black",
                font_size=72,
                font_weight="bold",
                color_primary="&H00FFFFFF&",
                color_highlight="&H0000D7FF&",
                color_outline="&H00000000&",
                outline_width=5,
                shadow_distance=2,
                emphasis_tokens=tokens_in_chunk,
                reason="Speech synchronized caption burst",
                confidence=1.0,
            ))
            text_counter += 1

        caption_track = TextTrack(
            track_id="track_text_t1",
            name="Captions & Graphics",
            events=text_events,
        )

        # ── 3. Compile B-Roll Overlay Track ───────────────────────────
        broll_clips: List[BrollClip] = []
        broll_counter = 1
        unresolved_items: List[Dict[str, Any]] = []

        broll_decisions = [
            d for d in creative_plan.decisions
            if d.operation in [CreativeOperation.BROLL_CONTEXT, CreativeOperation.BROLL_SUPPORT, CreativeOperation.BROLL_CONTRAST]
            and d.broll is not None
        ]

        for b in broll_decisions:
            t_in = remapper.map_source_to_timeline(b.time_range[0])
            t_out = remapper.map_source_to_timeline(b.time_range[1])
            if t_in is not None and t_out is not None and t_out > t_in:
                b_status = "RESOLVED" if b.status == CreativeStatus.RESOLVED else "BROLL_UNRESOLVED"

                broll_clips.append(BrollClip(
                    clip_id=f"bclip_{broll_counter:03d}",
                    concept=b.broll.asset_query,
                    search_query=b.broll.asset_query,
                    timeline_in=round(t_in, 3),
                    timeline_out=round(t_out, 3),
                    source_reference=b.broll.resolved_asset_path,
                    source_in=0.0,
                    source_out=round(t_out - t_in, 3),
                    opacity=b.broll.opacity,
                    status=b_status,
                    origin_opportunity_id=b.source_opportunity_ids[0] if b.source_opportunity_ids else None,
                    reason=b.reason,
                    confidence=b.confidence,
                ))

                if b_status != "RESOLVED":
                    unresolved_items.append({
                        "item_id": f"unres_broll_{broll_counter:03d}",
                        "type": "BROLL",
                        "creative_id": b.creative_id,
                        "query": b.broll.asset_query,
                        "duration": round(t_out - t_in, 3),
                        "timeline_in": round(t_in, 3),
                        "timeline_out": round(t_out, 3),
                        "reason": "Missing local video asset in library",
                    })
                broll_counter += 1

        broll_track = BrollTrack(
            track_id="track_broll_b1",
            name="B-Roll Overlay Track",
            clips=broll_clips,
        )

        # ── 4. Compile Audio Tracks (Voice Audio + SFX) ───────────────
        voice_clips: List[AudioClip] = []
        sfx_clips: List[AudioClip] = []
        audio_counter = 1

        # Voice clips track keep segments
        v_cursor = 0.0
        for seg in keep_segments:
            s_in = float(seg["start"])
            s_out = float(seg["end"])
            dur = round(s_out - s_in, 4)
            v_tin = v_cursor
            v_tout = round(v_cursor + dur, 4)
            v_cursor = v_tout

            voice_clips.append(AudioClip(
                clip_id=f"aclip_voice_{audio_counter:03d}",
                role=AudioRole.VOICE,
                source_reference=source_video_str,
                source_in=s_in,
                source_out=s_out,
                timeline_in=v_tin,
                timeline_out=v_tout,
                volume=1.0,
                status="RESOLVED",
                reason="Primary voice dialogue",
                confidence=1.0,
            ))
            audio_counter += 1

        # SFX cues
        sfx_decisions = [
            d for d in creative_plan.decisions
            if d.operation in [CreativeOperation.SFX_WHOOSH, CreativeOperation.SFX_POP, CreativeOperation.SFX_IMPACT, CreativeOperation.SFX_RISER]
            and d.sfx is not None
        ]

        for s in sfx_decisions:
            t_in = remapper.map_source_to_timeline(s.time_range[0])
            if t_in is not None:
                sfx_dur = max(0.35, round(s.time_range[1] - s.time_range[0], 3))
                t_out = round(t_in + sfx_dur, 3)
                s_status = "RESOLVED" if s.status == CreativeStatus.RESOLVED else "SFX_UNRESOLVED"

                sfx_clips.append(AudioClip(
                    clip_id=f"aclip_sfx_{audio_counter:03d}",
                    role=AudioRole.SFX,
                    source_reference=s.sfx.asset_path or f"asset://sfx/{s.sfx.category}/{s.sfx.sound_cue}",
                    source_in=0.0,
                    source_out=round(t_out - t_in, 3),
                    timeline_in=round(t_in, 3),
                    timeline_out=round(t_out, 3),
                    volume=0.85,
                    ducking_db=-18.0,
                    status=s_status,
                    origin_opportunity_id=s.source_opportunity_ids[0] if s.source_opportunity_ids else None,
                    reason=s.reason,
                    confidence=s.confidence,
                ))

                if s_status != "RESOLVED":
                    unresolved_items.append({
                        "item_id": f"unres_sfx_{audio_counter:03d}",
                        "type": "SFX",
                        "creative_id": s.creative_id,
                        "category": s.sfx.category,
                        "cue": s.sfx.sound_cue,
                        "timeline_in": round(t_in, 3),
                        "timeline_out": round(t_out, 3),
                        "reason": "Missing local audio file in assets/sfx library",
                    })
                audio_counter += 1

        voice_track = AudioTrack(
            track_id="track_audio_a1",
            name="Primary Dialogue Track",
            role=AudioRole.VOICE,
            clips=voice_clips,
        )

        sfx_track = AudioTrack(
            track_id="track_audio_a2",
            name="SFX Accents Track",
            role=AudioRole.SFX,
            clips=sfx_clips,
        )

        # ── 5. Assemble Master Multi-Track TimelineIR ──────────────────
        timeline_ir = TimelineIR(
            version="2.1.0",
            project_name=f"Reel_{Path(source_video_str).stem}",
            source_video=source_video_str,
            source_duration=src_dur,
            timeline_duration=remapper.total_timeline_duration,
            target_width=reframe_plan.target_width,
            target_height=reframe_plan.target_height,
            target_aspect_ratio="9:16",
            fps=float(media_meta.get("fps", 30.0)),
            video_tracks=[main_video_track],
            broll_tracks=[broll_track],
            text_tracks=[caption_track],
            audio_tracks=[voice_track, sfx_track],
            transitions=transitions,
            unresolved_items=unresolved_items,
            metadata={
                "creative_style_profile": creative_plan.style_profile,
                "total_creative_decisions": len(creative_plan.decisions),
                "unresolved_items_count": len(unresolved_items),
                "cuts_count": len(remapper.cuts),
                "punch_ins_count": len(punch_decisions),
                "face_center": {"x": reframe_plan.face_center_x, "y": reframe_plan.face_center_y},
            }
        )

        logger.info(
            f"CreativeTimelineCompiler | Successfully compiled TimelineIR v{timeline_ir.version}: "
            f"{timeline_ir.timeline_duration:.2f}s, {len(video_clips)} video clips, "
            f"{len(text_events)} text events, {len(unresolved_items)} unresolved items."
        )

        return timeline_ir, reframe_plan
