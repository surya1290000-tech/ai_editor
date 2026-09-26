"""
core/timeline/remapper.py

Authoritative Master Timecode Remapping System
─────────────────────────────────────────────
Provides deterministic bidirectional mapping between SOURCE TIME and EDITED TIMELINE TIME.

When cuts remove dead space, all subsequent events (captions, punch-ins, B-roll, audio cues)
shift cleanly by the exact cumulative trimmed duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class KeepInterval:
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float

    @property
    def duration(self) -> float:
        return round(self.source_end - self.source_start, 4)


class TimecodeRemapper:
    """
    Authoritative time-remapping coordinator.
    Builds monotonic piecewise-linear mapping from keep segments.
    """

    def __init__(self, keep_segments: List[dict]):
        """
        keep_segments: list of dicts with {"start": float, "end": float} in source time.
        """
        self.intervals: List[KeepInterval] = []
        self.cuts: List[dict] = []

        # Sort source intervals
        sorted_segs = sorted(keep_segments, key=lambda s: s["start"])
        timeline_cursor = 0.0

        for idx, seg in enumerate(sorted_segs):
            s_in = round(float(seg["start"]), 4)
            s_out = round(float(seg["end"]), 4)
            dur = round(s_out - s_in, 4)
            if dur <= 0.0:
                continue

            # Record cut if there is a gap before this segment
            if idx > 0:
                prev_out = self.intervals[-1].source_end
                if s_in > prev_out:
                    cut_dur = round(s_in - prev_out, 4)
                    self.cuts.append({
                        "cut_id": f"cut_{len(self.cuts)+1:03d}",
                        "source_start": prev_out,
                        "source_end": s_in,
                        "duration": cut_dur,
                        "timeline_boundary": timeline_cursor,
                    })

            t_in = round(timeline_cursor, 4)
            t_out = round(timeline_cursor + dur, 4)

            self.intervals.append(KeepInterval(
                source_start=s_in,
                source_end=s_out,
                timeline_start=t_in,
                timeline_end=t_out,
            ))
            timeline_cursor = t_out

        self.total_timeline_duration = round(timeline_cursor, 4)
        self.total_source_duration = self.intervals[-1].source_end if self.intervals else 0.0

    def map_source_to_timeline(self, source_sec: float, clamp_cuts: bool = False) -> Optional[float]:
        """
        Map a single source timestamp to edited timeline time.
        If source_sec falls inside a cut:
          - If clamp_cuts is False: returns None.
          - If clamp_cuts is True: returns the boundary timecode where the cut occurred.
        """
        source_sec = round(float(source_sec), 4)

        for interval in self.intervals:
            if interval.source_start <= source_sec <= interval.source_end:
                offset = source_sec - interval.source_start
                return round(interval.timeline_start + offset, 4)

        if not clamp_cuts:
            return None

        # Clamping logic for boundary events
        if not self.intervals:
            return 0.0
        if source_sec < self.intervals[0].source_start:
            return 0.0
        if source_sec > self.intervals[-1].source_end:
            return self.total_timeline_duration

        for cut in self.cuts:
            if cut["source_start"] <= source_sec <= cut["source_end"]:
                return cut["timeline_boundary"]

        return None

    def map_interval_to_timeline(
        self, source_start: float, source_end: float
    ) -> Optional[Tuple[float, float]]:
        """
        Map a source interval [source_start, source_end] to master timeline.
        Clamps to the active keep regions. Returns None if entire interval was cut.
        """
        t_start = self.map_source_to_timeline(source_start, clamp_cuts=True)
        t_end = self.map_source_to_timeline(source_end, clamp_cuts=True)

        if t_start is None or t_end is None:
            return None

        if t_end <= t_start:
            return None

        return (round(t_start, 4), round(t_end, 4))

    def map_timeline_to_source(self, timeline_sec: float) -> Optional[float]:
        """
        Inverse mapping: from edited timeline time to source video time.
        """
        timeline_sec = round(float(timeline_sec), 4)
        if timeline_sec < 0.0:
            return self.intervals[0].source_start if self.intervals else 0.0
        if timeline_sec > self.total_timeline_duration:
            return self.intervals[-1].source_end if self.intervals else 0.0

        for interval in self.intervals:
            if interval.timeline_start <= timeline_sec <= interval.timeline_end:
                offset = timeline_sec - interval.timeline_start
                return round(interval.source_start + offset, 4)

        return None

    def is_in_cut(self, source_sec: float) -> bool:
        """Returns True if the source timecode falls inside a trimmed dead-space pause."""
        return self.map_source_to_timeline(source_sec, clamp_cuts=False) is None
