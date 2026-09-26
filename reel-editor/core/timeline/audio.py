"""
core/timeline/audio.py

Multi-Track Audio Timeline Engine
─────────────────────────────────
Manages VOICE, MUSIC, SFX, and AMBIENCE tracks.
Ensures the speaker's voice remains dominant and tracks unresolved SFX cues
as placeholders without fabricating fake audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

from core.timeline.models import AudioTrack, AudioClip, AudioRole
from core.timeline.remapper import TimecodeRemapper


class AudioTimelineBuilder:
    """
    Builds the structured multi-track audio layout.
    """

    def __init__(self, sfx_dir: Optional[Path] = None):
        self.sfx_dir = sfx_dir

    def build_audio_tracks(
        self,
        source_audio_path: str,
        keep_segments: List[Dict[str, Any]],
        sound_requests: List[Dict[str, Any]],
        remapper: TimecodeRemapper,
    ) -> tuple[List[AudioTrack], List[Dict[str, Any]]]:
        """
        Builds Voice track and SFX track.
        """
        voice_clips: List[AudioClip] = []
        sfx_clips: List[AudioClip] = []
        unresolved_sfx: List[Dict[str, Any]] = []

        # ── 1. Main Voice Audio Track (Sliced exactly to keep segments)
        timeline_offset = 0.0
        for idx, seg in enumerate(keep_segments, start=1):
            s_in = float(seg["start"])
            s_out = float(seg["end"])
            dur = round(s_out - s_in, 4)
            t_in = timeline_offset
            t_out = round(timeline_offset + dur, 4)
            timeline_offset = t_out

            voice_clips.append(AudioClip(
                clip_id=f"voice_clip_{idx:03d}",
                role=AudioRole.VOICE,
                source_reference=source_audio_path,
                source_in=s_in,
                source_out=s_out,
                timeline_in=t_in,
                timeline_out=t_out,
                volume=1.0,
                fade_in_sec=0.015,
                fade_out_sec=0.015,
                status="RESOLVED",
                reason="Primary speaker dialogue",
                confidence=1.0,
            ))

        voice_track = AudioTrack(
            track_id="track_audio_a1",
            name="Primary Dialogue Track",
            role=AudioRole.VOICE,
            clips=voice_clips,
        )

        # ── 2. Sound Effects Track (SFX placeholders)
        for idx, sfx in enumerate(sound_requests, start=1):
            cue_type = sfx.get("cue_type", "whoosh")
            src_time = float(sfx.get("timestamp", 0.0))
            t_mapped = remapper.map_source_to_timeline(src_time, clamp_cuts=True)
            if t_mapped is None:
                continue

            # Check if local SFX file exists
            resolved_path = None
            if self.sfx_dir and self.sfx_dir.exists():
                candidates = list(self.sfx_dir.glob(f"*{cue_type}*.wav"))
                if candidates:
                    resolved_path = str(candidates[0])

            if resolved_path:
                sfx_clips.append(AudioClip(
                    clip_id=f"sfx_{idx:03d}",
                    role=AudioRole.SFX,
                    source_reference=resolved_path,
                    source_in=0.0,
                    source_out=0.5,
                    timeline_in=t_mapped,
                    timeline_out=round(t_mapped + 0.5, 3),
                    volume=0.35,
                    status="RESOLVED",
                    origin_opportunity_id=sfx.get("origin_opportunity_id"),
                    reason=sfx.get("reason", "Transition sound accent"),
                    confidence=float(sfx.get("confidence", 0.8)),
                ))
            else:
                sfx_clips.append(AudioClip(
                    clip_id=f"sfx_placeholder_{idx:03d}",
                    role=AudioRole.SFX,
                    source_reference="",
                    source_in=0.0,
                    source_out=0.4,
                    timeline_in=t_mapped,
                    timeline_out=round(t_mapped + 0.4, 3),
                    volume=0.0,
                    status="SFX_UNRESOLVED",
                    origin_opportunity_id=sfx.get("origin_opportunity_id"),
                    reason=f"{sfx.get('reason', '')} (SFX cue '{cue_type}' unresolved — voice remains pure)",
                    confidence=float(sfx.get("confidence", 0.8)),
                ))
                unresolved_sfx.append({
                    "item_type": "SFX_CUE",
                    "cue_type": cue_type,
                    "timeline_time": t_mapped,
                    "status": "SFX_UNRESOLVED",
                    "action_taken": "No audio artifact injected; dialogue preserved with 100% clarity",
                })

        sfx_track = AudioTrack(
            track_id="track_audio_a2",
            name="Sound Accents Track",
            role=AudioRole.SFX,
            clips=sfx_clips,
        )

        return [voice_track, sfx_track], unresolved_sfx
