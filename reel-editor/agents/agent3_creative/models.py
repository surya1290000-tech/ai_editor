"""
agents/agent3_creative/models.py

Creative Decision Layer Models & Contracts (Agent 3)
────────────────────────────────────────────────────
Formal typed data contracts specifying HOW approved editorial decisions
are executed visually and acoustically.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


class CreativeOperation(str, Enum):
    KEEP = "KEEP"
    NO_EDIT = "NO_EDIT"
    REMOVE_FILLER = "REMOVE_FILLER"
    REMOVE_DEAD_SPACE = "REMOVE_DEAD_SPACE"
    PACE_TRIM = "PACE_TRIM"
    CUT = "CUT"
    SOFT_CUT = "SOFT_CUT"
    PUNCH_IN = "PUNCH_IN"
    EMPHASIS_TEXT = "EMPHASIS_TEXT"
    HERO_TEXT = "HERO_TEXT"
    EDITORIAL_CARD = "EDITORIAL_CARD"
    BROLL_SUPPORT = "BROLL_SUPPORT"
    BROLL_CONTEXT = "BROLL_CONTEXT"
    BROLL_CONTRAST = "BROLL_CONTRAST"
    SFX_WHOOSH = "SFX_WHOOSH"
    SFX_POP = "SFX_POP"
    SFX_IMPACT = "SFX_IMPACT"
    SFX_RISER = "SFX_RISER"
    MUSIC_ENTER = "MUSIC_ENTER"
    MUSIC_EXIT = "MUSIC_EXIT"
    AUDIO_DUCK = "AUDIO_DUCK"


class CreativeStatus(str, Enum):
    RESOLVED = "RESOLVED"
    UNRESOLVED_ASSET = "UNRESOLVED_ASSET"
    CONFLICT = "CONFLICT"
    PRESERVED = "PRESERVED"


@dataclass
class TypographySpec:
    font: str
    font_fallback: list[str]
    size: int
    weight: str = "bold"
    position: str = "center"          # "top", "center", "lower_third"
    safe_zone: bool = True
    line_breaking: str = "balanced"   # "balanced", "auto"
    color_primary: str = "&H00FFFFFF&"
    color_highlight: str = "&H0000D7FF&"
    color_outline: str = "&H00000000&"
    outline_width: int = 5
    shadow_distance: int = 2
    vertical_line_padding: float = 1.0
    is_indic_script: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypographySpec:
        return cls(**data)


@dataclass
class AnimationSpec:
    preset: str = "pop"              # "pop", "fade", "overshoot", "spring"
    easing: str = "ease_out"         # "linear", "ease_in", "ease_out", "ease_in_out", "spring", "pop", "overshoot", "back"
    duration: float = 0.28
    primitives: dict[str, Any] = field(default_factory=lambda: {
        "opacity": [0.0, 1.0],
        "scale": [0.94, 1.0],
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnimationSpec:
        return cls(**data)


@dataclass
class PunchInSpec:
    scale: float
    anchor: dict[str, float]         # {"x": 0.42, "y": 0.43}
    transition: str = "JUMP_CUT"
    conceptual_strength: str = "MEDIUM"
    headroom_preserved: bool = True
    duration: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PunchInSpec:
        return cls(**data)


@dataclass
class BrollSpec:
    asset_query: str
    role: str = "context"             # "context", "support", "contrast"
    required_duration: float = 3.0
    aspect_treatment: str = "9:16_FILL_CROP"
    crop: Optional[dict[str, float]] = None
    overlay_treatment: str = "FULL_OVERLAY"
    transition: str = "CROSSFADE"
    opacity: float = 1.0
    replacement_mode: str = "OVERLAY"  # "OVERLAY" or "CUTAWAY"
    status: str = "UNRESOLVED_ASSET"
    resolved_asset_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrollSpec:
        return cls(**data)


@dataclass
class SfxSpec:
    category: str                    # "whoosh", "pop", "impact", "riser", "subtle_hit"
    sound_cue: str
    asset_path: Optional[str] = None
    start: float = 0.0
    end: float = 0.0
    gain_db: float = -15.0
    fade_in_sec: float = 0.02
    fade_out_sec: float = 0.04
    status: str = "UNRESOLVED_ASSET"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SfxSpec:
        return cls(**data)


@dataclass
class AudioSpec:
    voice_priority: float = 1.0
    music_level: float = 0.0
    sfx_level: float = 0.85
    ducking_profile: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "duck_db": -18.0,
        "attack_ms": 20,
        "release_ms": 250,
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioSpec:
        return cls(**data)


@dataclass
class CreativeDecision:
    """
    Concrete Creative Decision produced by Agent 3.
    Specifies HOW an approved Agent 2 editorial intervention is realized.
    """
    creative_id: str
    source_decision_id: str
    operation: CreativeOperation
    time_range: list[float]          # [start, end]
    style_profile: str
    status: CreativeStatus = CreativeStatus.RESOLVED
    confidence: float = 1.0
    reason: str = ""
    story_act: str = "HOOK"
    source_opportunity_ids: list[str] = field(default_factory=list)
    source_evidence_ids: list[str] = field(default_factory=list)

    # Specific treatment payloads (only present when relevant)
    text_content: Optional[str] = None
    typography: Optional[TypographySpec] = None
    animation: Optional[AnimationSpec] = None
    punch_in: Optional[PunchInSpec] = None
    broll: Optional[BrollSpec] = None
    sfx: Optional[SfxSpec] = None
    audio: Optional[AudioSpec] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "creative_id": self.creative_id,
            "source_decision_id": self.source_decision_id,
            "operation": self.operation.value if isinstance(self.operation, CreativeOperation) else self.operation,
            "time_range": [round(self.time_range[0], 3), round(self.time_range[1], 3)],
            "style_profile": self.style_profile,
            "status": self.status.value if isinstance(self.status, CreativeStatus) else self.status,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "story_act": self.story_act,
            "source_opportunity_ids": self.source_opportunity_ids,
            "source_evidence_ids": self.source_evidence_ids,
        }
        if self.text_content:
            d["text_content"] = self.text_content
        if self.typography:
            d["typography"] = self.typography.to_dict()
        if self.animation:
            d["animation"] = self.animation.to_dict()
        if self.punch_in:
            d["punch_in"] = self.punch_in.to_dict()
        if self.broll:
            d["broll"] = self.broll.to_dict()
        if self.sfx:
            d["sfx"] = self.sfx.to_dict()
        if self.audio:
            d["audio"] = self.audio.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreativeDecision:
        d = dict(data)
        op = CreativeOperation(d["operation"])
        stat = CreativeStatus(d.get("status", "RESOLVED"))
        typo = TypographySpec.from_dict(d["typography"]) if d.get("typography") else None
        anim = AnimationSpec.from_dict(d["animation"]) if d.get("animation") else None
        punch = PunchInSpec.from_dict(d["punch_in"]) if d.get("punch_in") else None
        broll = BrollSpec.from_dict(d["broll"]) if d.get("broll") else None
        sfx = SfxSpec.from_dict(d["sfx"]) if d.get("sfx") else None
        aud = AudioSpec.from_dict(d["audio"]) if d.get("audio") else None

        return cls(
            creative_id=d["creative_id"],
            source_decision_id=d["source_decision_id"],
            operation=op,
            time_range=d["time_range"],
            style_profile=d.get("style_profile", "EDITORIAL_CINEMATIC"),
            status=stat,
            confidence=float(d.get("confidence", 1.0)),
            reason=d.get("reason", ""),
            story_act=d.get("story_act", "HOOK"),
            source_opportunity_ids=d.get("source_opportunity_ids", []),
            source_evidence_ids=d.get("source_evidence_ids", []),
            text_content=d.get("text_content"),
            typography=typo,
            animation=anim,
            punch_in=punch,
            broll=broll,
            sfx=sfx,
            audio=aud,
        )


@dataclass
class CreativePlan:
    """
    Master container for Agent 3's creative output plan.
    """
    version: str = "4.0.0"
    source_video: str = ""
    duration_seconds: float = 0.0
    style_profile: str = "EDITORIAL_CINEMATIC"
    total_decisions: int = 0
    resolved_count: int = 0
    unresolved_assets_count: int = 0
    conflict_count: int = 0
    preserved_spans_count: int = 0
    decisions: list[CreativeDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.decisions:
            self.total_decisions = len(self.decisions)
            self.resolved_count = sum(1 for d in self.decisions if d.status == CreativeStatus.RESOLVED)
            self.unresolved_assets_count = sum(1 for d in self.decisions if d.status == CreativeStatus.UNRESOLVED_ASSET)
            self.conflict_count = sum(1 for d in self.decisions if d.status == CreativeStatus.CONFLICT)
            self.preserved_spans_count = sum(1 for d in self.decisions if d.status == CreativeStatus.PRESERVED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "duration_seconds": round(self.duration_seconds, 3),
            "style_profile": self.style_profile,
            "total_decisions": len(self.decisions),
            "resolved_count": sum(1 for d in self.decisions if d.status == CreativeStatus.RESOLVED),
            "unresolved_assets_count": sum(1 for d in self.decisions if d.status == CreativeStatus.UNRESOLVED_ASSET),
            "conflict_count": sum(1 for d in self.decisions if d.status == CreativeStatus.CONFLICT),
            "preserved_spans_count": sum(1 for d in self.decisions if d.status == CreativeStatus.PRESERVED),
            "decisions": [d.to_dict() for d in self.decisions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreativePlan:
        return cls(
            version=data.get("version", "4.0.0"),
            source_video=data.get("source_video", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            style_profile=data.get("style_profile", "EDITORIAL_CINEMATIC"),
            total_decisions=data.get("total_decisions", 0),
            resolved_count=data.get("resolved_count", 0),
            unresolved_assets_count=data.get("unresolved_assets_count", 0),
            conflict_count=data.get("conflict_count", 0),
            preserved_spans_count=data.get("preserved_spans_count", 0),
            decisions=[CreativeDecision.from_dict(d) for d in data.get("decisions", [])],
            metadata=data.get("metadata", {}),
        )
