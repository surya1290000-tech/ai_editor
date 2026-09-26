"""
core/timeline/captions.py

Kinetic Caption Compiler
────────────────────────
Compiles word-level Whisper transcript into master-timeline synchronized,
high-contrast, mobile-optimized ASS kinetic subtitles with vibrant keyword highlights.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.timeline.models import TextTrack, TextEvent
from core.timeline.remapper import TimecodeRemapper


class CaptionCompiler:
    """
    Compiles word timestamps through the authoritative TimecodeRemapper
    into frame-accurate styled ASS subtitle events.
    """

    def __init__(
        self,
        remapper: TimecodeRemapper,
        target_width: int = 1080,
        target_height: int = 1920,
        font_name: str = "Arial Black",
        font_size: int = 72,
        primary_color: str = "&H00FFFFFF&",    # White
        highlight_color: str = "&H0000D7FF&",  # Bright Yellow (ASS hex &HAABBGGRR&)
        outline_color: str = "&H00000000&",    # Pure Black
        outline_width: int = 5,
        shadow_distance: int = 2,
        margin_v: int = 380,                   # Safe-zone lower third
    ):
        self.remapper = remapper
        self.target_width = target_width
        self.target_height = target_height
        self.font_name = font_name
        self.font_size = font_size
        self.primary_color = primary_color
        self.highlight_color = highlight_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.shadow_distance = shadow_distance
        self.margin_v = margin_v

    def compile(
        self,
        transcript_raw: Dict[str, Any],
        fused_timeline: Optional[Dict[str, Any]] = None,
        emphasis_directives: Optional[List[Dict[str, Any]]] = None,
    ) -> TextTrack:
        """
        Takes raw word transcripts, maps them to master timeline, and builds TextTrack with TextEvents.
        """
        emphasis_directives = emphasis_directives or []
        emphasis_lookup = {
            re.sub(r"[^\w]", "", str(d.get("word", ""))).lower()
            for d in emphasis_directives
            if d.get("word")
        }

        # 1. Gather all raw words
        words = transcript_raw.get("words", [])
        if not words and fused_timeline:
            for s in fused_timeline.get("sentences", []):
                words.extend(s.get("words", []))

        # 2. Map words through authoritative TimecodeRemapper
        mapped_words = []
        for w in words:
            s_in = float(w.get("start", 0.0))
            s_out = float(w.get("end", 0.0))

            t_in = self.remapper.map_source_to_timeline(s_in)
            t_out = self.remapper.map_source_to_timeline(s_out)

            if t_in is not None and t_out is not None and t_out > t_in:
                clean_text = str(w.get("word", "")).strip()
                if clean_text:
                    mapped_words.append({
                        "word": clean_text,
                        "source_start": s_in,
                        "source_end": s_out,
                        "timeline_in": t_in,
                        "timeline_out": t_out,
                        "emphasis_score": float(w.get("emphasis_score") or 0.0),
                    })

        # 3. Group into snappy 2 to 4 word bursts
        chunks: List[List[Dict[str, Any]]] = []
        curr_chunk: List[Dict[str, Any]] = []

        for w in mapped_words:
            curr_chunk.append(w)
            has_break = any(p in w["word"] for p in [".", ",", "?", "!", ";"])
            if len(curr_chunk) >= 3 or has_break:
                chunks.append(curr_chunk)
                curr_chunk = []
        if curr_chunk:
            chunks.append(curr_chunk)

        # 4. Build TextEvent objects
        events: List[TextEvent] = []
        for idx, chunk in enumerate(chunks, start=1):
            c_start = chunk[0]["timeline_in"]
            c_end = chunk[-1]["timeline_out"]

            # Identify emphasized words in this chunk
            chunk_emphasis: List[str] = []
            words_formatted: List[str] = []

            for w in chunk:
                clean_tok = re.sub(r"[^\w]", "", w["word"]).lower()
                is_emp = (clean_tok in emphasis_lookup) or (w["emphasis_score"] >= 0.62)

                if is_emp:
                    chunk_emphasis.append(w["word"])
                    # Styled with highlight tag
                    words_formatted.append(f"{{\\c{self.highlight_color}}}{w['word'].upper()}{{\\c{self.primary_color}}}")
                else:
                    words_formatted.append(w["word"].upper())

            formatted_text = " ".join(words_formatted)

            events.append(TextEvent(
                event_id=f"cap_{idx:03d}",
                text=formatted_text,
                timeline_in=c_start,
                timeline_out=c_end,
                source_start=chunk[0]["source_start"],
                source_end=chunk[-1]["source_end"],
                margin_v=self.margin_v,
                font_name=self.font_name,
                font_size=self.font_size,
                color_primary=self.primary_color,
                color_highlight=self.highlight_color,
                color_outline=self.outline_color,
                outline_width=self.outline_width,
                shadow_distance=self.shadow_distance,
                emphasis_tokens=chunk_emphasis,
                origin_opportunity_id=f"opp_caption_{idx:03d}",
                reason="Kinetic subtitle burst aligned to speech cadence",
                confidence=1.0,
            ))

        return TextTrack(track_id="track_captions_v2", name="Kinetic Captions", events=events)

    def write_ass_file(self, text_track: TextTrack, output_path: Path):
        """Generates a valid, complete ASS file ready for FFmpeg libass."""
        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {self.target_width}
PlayResY: {self.target_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ReelStyle,{self.font_name},{self.font_size},{self.primary_color},&H000000FF&,{self.outline_color},&H80000000&,-1,0,0,0,100,100,0,0,1,{self.outline_width},{self.shadow_distance},2,50,50,{self.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogues = []
        for ev in text_track.events:
            t_start = self._seconds_to_ass(ev.timeline_in)
            t_end = self._seconds_to_ass(ev.timeline_out)
            dialogues.append(f"Dialogue: 0,{t_start},{t_end},ReelStyle,,0,0,0,,{ev.text}")

        output_path.write_text(ass_header + "\n".join(dialogues) + "\n", encoding="utf-8")

    def write_srt_file(self, text_track: TextTrack, output_path: Path):
        """Generates standard SRT subtitles for DaVinci Resolve import."""
        lines = []
        for idx, ev in enumerate(text_track.events, start=1):
            t_start = self._seconds_to_srt(ev.timeline_in)
            t_end = self._seconds_to_srt(ev.timeline_out)
            # Strip ASS color formatting tags for clean plain text SRT
            clean_text = re.sub(r"\{.*?\}", "", ev.text)
            lines.append(f"{idx}")
            lines.append(f"{t_start} --> {t_end}")
            lines.append(clean_text)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _seconds_to_ass(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

    @staticmethod
    def _seconds_to_srt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
