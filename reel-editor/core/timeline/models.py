"""
core/timeline/models.py

Timeline Intermediate Representation (Timeline IR)
───────────────────────────────────────────────────
The definitive, deterministic data contract between creative decision-making
(Agent 2) and video execution / rendering (Agent 3 & Render Compiler).

Every operation retains:
- source_reference
- reason
- confidence
- origin_opportunity_id
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Optional, Any, List, Dict


class TrackType(str, Enum):
    VIDEO = "VIDEO"
    TEXT = "TEXT"
    BROLL = "BROLL"
    AUDIO = "AUDIO"


class AudioRole(str, Enum):
    VOICE = "VOICE"
    MUSIC = "MUSIC"
    SFX = "SFX"
    AMBIENCE = "AMBIENCE"


@dataclass
class CropRect:
    x: float = 0.0      # Normalized 0..1 or pixel coordinate
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    @property
    def width(self) -> float:
        return self.w

    @property
    def height(self) -> float:
        return self.h

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict) -> CropRect:
        if not data:
            return cls()
        w = data.get("w", data.get("width", 1.0))
        h = data.get("h", data.get("height", 1.0))
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0), w=w, h=h)



@dataclass
class Transform2D:
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Transform2D:
        return cls(**data) if data else cls()


@dataclass
class VideoClip:
    clip_id: str
    source_reference: str
    source_in: float
    source_out: float
    timeline_in: float
    timeline_out: float
    crop: Optional[CropRect] = None
    scale: float = 1.0
    position: Transform2D = field(default_factory=Transform2D)
    opacity: float = 1.0
    is_punch_in: bool = False
    origin_opportunity_id: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return round(self.timeline_out - self.timeline_in, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.crop:
            d["crop"] = self.crop.to_dict()
        d["position"] = self.position.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> VideoClip:
        d = dict(data)
        if "source_reference" not in d and "source_path" in d:
            d["source_reference"] = d.pop("source_path")
        if d.get("crop"):
            d["crop"] = CropRect.from_dict(d["crop"])
        if d.get("position"):
            d["position"] = Transform2D.from_dict(d["position"])
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class TextEvent:
    event_id: str
    text: str
    timeline_in: float
    timeline_out: float
    source_start: float = 0.0
    source_end: float = 0.0
    position_align: int = 2          # ASS standard: 2 = bottom-center, 5 = mid-center
    margin_v: int = 380              # Vertical offset in target coordinates
    font_name: str = "Arial Black"
    font_size: int = 72
    font_weight: str = "bold"
    color_primary: str = "&H00FFFFFF&"    # ASS format: &HAABBGGRR& (White)
    color_highlight: str = "&H0000D7FF&"  # ASS format: &HAABBGGRR& (Gold/Yellow)
    color_outline: str = "&H00000000&"    # ASS format: &HAABBGGRR& (Black)
    outline_width: int = 5
    shadow_distance: int = 2
    emphasis_tokens: list[str] = field(default_factory=list)
    origin_opportunity_id: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return round(self.timeline_out - self.timeline_in, 3)

    @property
    def start_timeline(self) -> float:
        return self.timeline_in

    @property
    def end_timeline(self) -> float:
        return self.timeline_out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TextEvent:
        d = dict(data)
        if "text" not in d and "content" in d:
            d["text"] = d.pop("content")
        if "font_name" not in d and "font_family" in d:
            d["font_name"] = d.pop("font_family")
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class BrollClip:
    clip_id: str
    concept: str
    search_query: str
    timeline_in: float
    timeline_out: float
    source_reference: Optional[str] = None
    source_in: float = 0.0
    source_out: float = 0.0
    opacity: float = 1.0
    scale: float = 1.0
    crop: Optional[CropRect] = None
    status: str = "BROLL_UNRESOLVED"      # "RESOLVED" or "BROLL_UNRESOLVED"
    origin_opportunity_id: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return round(self.timeline_out - self.timeline_in, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.crop:
            d["crop"] = self.crop.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> BrollClip:
        d = dict(data)
        if d.get("crop"):
            d["crop"] = CropRect.from_dict(d["crop"])
        return cls(**d)


@dataclass
class AudioClip:
    clip_id: str
    role: AudioRole
    source_reference: str
    source_in: float
    source_out: float
    timeline_in: float
    timeline_out: float
    volume: float = 1.0
    ducking_db: float = 0.0
    fade_in_sec: float = 0.02
    fade_out_sec: float = 0.02
    status: str = "RESOLVED"             # "RESOLVED" or "SFX_UNRESOLVED"
    origin_opportunity_id: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return round(self.timeline_out - self.timeline_in, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value if isinstance(self.role, AudioRole) else self.role
        return d

    @classmethod
    def from_dict(cls, data: dict) -> AudioClip:
        d = dict(data)
        if "role" not in d:
            d["role"] = AudioRole.SFX
        elif isinstance(d["role"], str):
            d["role"] = AudioRole(d["role"]) if d["role"] in [r.value for r in AudioRole] else AudioRole.SFX
        if "source_reference" not in d and "source_path" in d:
            d["source_reference"] = d.pop("source_path")
        elif "source_reference" not in d:
            d["source_reference"] = ""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class Transition:
    transition_id: str
    transition_type: str = "HARD_CUT"    # "HARD_CUT", "CROSSFADE", "WHIP_PAN"
    timeline_time: float = 0.0
    duration: float = 0.0
    outgoing_clip_id: str = ""
    incoming_clip_id: str = ""
    origin_opportunity_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Transition:
        return cls(**data)


@dataclass
class VideoTrack:
    track_id: str
    name: str = "Main Video"
    clips: list[VideoClip] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict) -> VideoTrack:
        return cls(
            track_id=data["track_id"],
            name=data.get("name", "Video Track"),
            clips=[VideoClip.from_dict(c) for c in data.get("clips", [])],
        )


@dataclass
class TextTrack:
    track_id: str
    name: str = "Captions"
    events: list[TextEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TextTrack:
        return cls(
            track_id=data["track_id"],
            name=data.get("name", "Text Track"),
            events=[TextEvent.from_dict(e) for e in data.get("events", [])],
        )


@dataclass
class BrollTrack:
    track_id: str
    name: str = "B-Roll Overlay"
    clips: list[BrollClip] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict) -> BrollTrack:
        return cls(
            track_id=data["track_id"],
            name=data.get("name", "B-Roll Track"),
            clips=[BrollClip.from_dict(c) for c in data.get("clips", [])],
        )


@dataclass
class AudioTrack:
    track_id: str
    name: str = "Voice Audio"
    role: AudioRole = AudioRole.VOICE
    clips: list[AudioClip] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "role": self.role.value if isinstance(self.role, AudioRole) else self.role,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, data: dict) -> AudioTrack:
        return cls(
            track_id=data["track_id"],
            name=data.get("name", "Audio Track"),
            role=AudioRole(data.get("role", "VOICE")),
            clips=[AudioClip.from_dict(c) for c in data.get("clips", [])],
        )


@dataclass
class TimelineIR:
    """
    The complete, authoritative multi-track Timeline Intermediate Representation.
    """
    version: str = "2.0.0"
    project_name: str = "Reel Project"
    source_video: str = ""
    source_duration: float = 0.0
    timeline_duration: float = 0.0
    target_width: int = 1080
    target_height: int = 1920
    target_aspect_ratio: str = "9:16"
    fps: float = 30.0

    # Multi-track structure
    video_tracks: list[VideoTrack] = field(default_factory=list)
    broll_tracks: list[BrollTrack] = field(default_factory=list)
    text_tracks: list[TextTrack] = field(default_factory=list)
    audio_tracks: list[AudioTrack] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)

    # Metadata & auditing
    unresolved_items: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_main_video_track(self) -> Optional[VideoTrack]:
        return self.video_tracks[0] if self.video_tracks else None

    def get_captions_track(self) -> Optional[TextTrack]:
        return self.text_tracks[0] if self.text_tracks else None

    def get_voice_audio_track(self) -> Optional[AudioTrack]:
        for t in self.audio_tracks:
            if t.role == AudioRole.VOICE:
                return t
        return None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "project_name": self.project_name,
            "source_video": self.source_video,
            "source_duration": round(self.source_duration, 3),
            "timeline_duration": round(self.timeline_duration, 3),
            "target_width": self.target_width,
            "target_height": self.target_height,
            "target_aspect_ratio": self.target_aspect_ratio,
            "fps": self.fps,
            "video_tracks": [t.to_dict() for t in self.video_tracks],
            "broll_tracks": [t.to_dict() for t in self.broll_tracks],
            "text_tracks": [t.to_dict() for t in self.text_tracks],
            "audio_tracks": [t.to_dict() for t in self.audio_tracks],
            "transitions": [t.to_dict() for t in self.transitions],
            "unresolved_items": self.unresolved_items,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimelineIR:
        return cls(
            version=data.get("version", "2.0.0"),
            project_name=data.get("project_name", "Reel Project"),
            source_video=data.get("source_video", ""),
            source_duration=data.get("source_duration", 0.0),
            timeline_duration=data.get("timeline_duration", 0.0),
            target_width=data.get("target_width", 1080),
            target_height=data.get("target_height", 1920),
            target_aspect_ratio=data.get("target_aspect_ratio", "9:16"),
            fps=data.get("fps", 30.0),
            video_tracks=[VideoTrack.from_dict(t) for t in data.get("video_tracks", [])],
            broll_tracks=[BrollTrack.from_dict(t) for t in data.get("broll_tracks", [])],
            text_tracks=[TextTrack.from_dict(t) for t in data.get("text_tracks", [])],
            audio_tracks=[AudioTrack.from_dict(t) for t in data.get("audio_tracks", [])],
            transitions=[Transition.from_dict(t) for t in data.get("transitions", [])],
            unresolved_items=data.get("unresolved_items", []),
            metadata=data.get("metadata", {}),
        )
