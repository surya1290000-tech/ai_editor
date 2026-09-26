"""
core/evidence/models.py

Standardized Evidence & Observation Layer Data Contracts (Phase 2)
─────────────────────────────────────────────────────────────────
Defines the typed data structures for:
  1. Enriched Transcript (with per-word confidence, code-switching, and uncertainty)
  2. Standardized Evidence Items (evidence_inventory.json)
  3. Story Observations (story_observations.json)
  4. Opportunity Inventory (opportunity_inventory.json)

Core Architectural Principles:
  - Agent 1 is an OBSERVER answering "What is happening in this footage?"
  - Strict distinction: OBSERVATION vs INTERPRETATION vs EDITORIAL DECISION
  - NO creative decisions (no font choices, no zoom values, no styling, no SFX tracks)
  - NO_EDIT is a first-class citizen
  - Low-confidence ASR propagates uncertainty to downstream semantics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


SUPPORTED_EVIDENCE_TYPES = (
    "hook_candidate",
    "sentence_boundary",
    "rhetorical_pause",
    "filler",
    "dead_space",
    "emphasis",
    "emotional_change",
    "energy_change",
    "topic_change",
    "narrative_transition",
    "contradiction",
    "realization",
    "payoff",
    "repeated_point",
    "weak_delivery",
    "strong_delivery",
    "visual_change",
    "framing_change",
    "face_position_change",
    "motion_change",
    "scene_boundary",
    "broll_opportunity",
    "punch_in_opportunity",
    "sfx_opportunity",
    "no_edit_region",
)

STORY_ACT_TYPES = (
    "HOOK",
    "CONTEXT",
    "PROBLEM",
    "BUILD_UP",
    "EXAMPLE",
    "CONTRAST",
    "REALIZATION",
    "PAYOFF",
    "CTA",
    "ENDING",
)

OPPORTUNITY_TYPES = (
    "punch_in_opportunity",
    "broll_opportunity",
    "sfx_opportunity",
    "cut_opportunity",
    "pace_trim_opportunity",
    "emphasis_text_opportunity",
    "hero_text_opportunity",
    "editorial_card_opportunity",
    "no_edit_opportunity",
)


# ─── Enriched Transcript Models ───────────────────────────────────────────────

@dataclass
class EnrichedWord:
    """Word with acoustic, language, visual context and uncertainty flags."""
    id: int
    word: str
    start: float
    end: float
    duration: float
    confidence: float
    language: str
    script: str
    romanized: str
    is_filler: bool = False
    emphasis_score: float = 0.0
    energy_deviation: float = 0.0
    pitch_deviation: float = 0.0
    low_confidence_flag: bool = False
    alternate_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrichedWord:
        return cls(
            id=data["id"],
            word=data["word"],
            start=float(data["start"]),
            end=float(data["end"]),
            duration=float(data.get("duration", data["end"] - data["start"])),
            confidence=float(data["confidence"]),
            language=data.get("language", "und"),
            script=data.get("script", "latin"),
            romanized=data.get("romanized", data["word"]),
            is_filler=bool(data.get("is_filler", False)),
            emphasis_score=float(data.get("emphasis_score", 0.0)),
            energy_deviation=float(data.get("energy_deviation", 0.0)),
            pitch_deviation=float(data.get("pitch_deviation", 0.0)),
            low_confidence_flag=bool(data.get("low_confidence_flag", False)),
            alternate_hypotheses=data.get("alternate_hypotheses", []),
            uncertainty=data.get("uncertainty", []),
        )


@dataclass
class EnrichedSentence:
    """Sentence with multimodal context and uncertainty aggregation."""
    id: int
    text: str
    start: float
    end: float
    duration: float
    language_mix: list[str] = field(default_factory=list)
    word_ids: list[int] = field(default_factory=list)
    speaking_rate_wpm: float = 0.0
    avg_energy_deviation: float = 0.0
    peak_energy_deviation: float = 0.0
    avg_confidence: float = 1.0
    low_confidence_count: int = 0
    filler_count: int = 0
    top_emphasis_word: Optional[dict[str, Any]] = None
    visual_context: dict[str, Any] = field(default_factory=dict)
    uncertainty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrichedSentence:
        return cls(
            id=data["id"],
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            duration=float(data.get("duration", data["end"] - data["start"])),
            language_mix=data.get("language_mix", []),
            word_ids=data.get("word_ids", []),
            speaking_rate_wpm=float(data.get("speaking_rate_wpm", 0.0)),
            avg_energy_deviation=float(data.get("avg_energy_deviation", 0.0)),
            peak_energy_deviation=float(data.get("peak_energy_deviation", 0.0)),
            avg_confidence=float(data.get("avg_confidence", 1.0)),
            low_confidence_count=int(data.get("low_confidence_count", 0)),
            filler_count=int(data.get("filler_count", 0)),
            top_emphasis_word=data.get("top_emphasis_word"),
            visual_context=data.get("visual_context", {}),
            uncertainty=data.get("uncertainty", []),
        )


@dataclass
class EnrichedTranscript:
    """Full enriched transcript preserving multilingual details and uncertainty."""
    version: str = "3.0.0"
    source_audio: str = ""
    duration_seconds: float = 0.0
    primary_language: str = "und"
    detected_languages: list[str] = field(default_factory=list)
    words: list[EnrichedWord] = field(default_factory=list)
    sentences: list[EnrichedSentence] = field(default_factory=list)
    speech_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_audio": self.source_audio,
            "duration_seconds": self.duration_seconds,
            "primary_language": self.primary_language,
            "detected_languages": self.detected_languages,
            "words": [w.to_dict() for w in self.words],
            "sentences": [s.to_dict() for s in self.sentences],
            "speech_stats": self.speech_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnrichedTranscript:
        return cls(
            version=data.get("version", "3.0.0"),
            source_audio=data.get("source_audio", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            primary_language=data.get("primary_language", "und"),
            detected_languages=data.get("detected_languages", []),
            words=[EnrichedWord.from_dict(w) for w in data.get("words", [])],
            sentences=[EnrichedSentence.from_dict(s) for s in data.get("sentences", [])],
            speech_stats=data.get("speech_stats", {}),
        )


# ─── Standardized Evidence Item Models ────────────────────────────────────────

@dataclass
class EvidenceSource:
    """Grounding references to transcript segments and words."""
    transcript_segment_ids: list[int] = field(default_factory=list)
    word_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceSource:
        return cls(
            transcript_segment_ids=list(data.get("transcript_segment_ids", [])),
            word_ids=list(data.get("word_ids", [])),
        )


@dataclass
class EvidenceSignals:
    """Multi-modal signal measurements underpinning this observation."""
    speech: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    visual: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceSignals:
        return cls(
            speech=data.get("speech", {}),
            audio=data.get("audio", {}),
            visual=data.get("visual", {}),
        )


@dataclass
class EvidenceItem:
    """
    Standardized atomic evidence unit.
    Captures WHAT is happening in the footage with measured multi-modal signals.
    Does NOT contain editing commands or creative styling.
    """
    evidence_id: str
    type: str
    start: float
    end: float
    source: EvidenceSource
    observation: str
    signals: EvidenceSignals
    confidence: float
    reliability: str  # "high" | "medium" | "low"
    uncertainty: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.type not in SUPPORTED_EVIDENCE_TYPES:
            raise ValueError(
                f"Invalid evidence type '{self.type}'. Supported types: {SUPPORTED_EVIDENCE_TYPES}"
            )
        if self.reliability not in ("high", "medium", "low"):
            raise ValueError(f"Invalid reliability '{self.reliability}'. Must be high, medium, or low.")
        if self.start < 0.0 or self.end < self.start:
            raise ValueError(f"Invalid time range: start={self.start}, end={self.end}")
        self.confidence = max(0.0, min(1.0, round(float(self.confidence), 3)))
        self.start = round(float(self.start), 3)
        self.end = round(float(self.end), 3)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "source": self.source.to_dict(),
            "observation": self.observation,
            "signals": self.signals.to_dict(),
            "confidence": self.confidence,
            "reliability": self.reliability,
            "uncertainty": self.uncertainty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceItem:
        src = data.get("source", {})
        source = EvidenceSource.from_dict(src) if isinstance(src, dict) else EvidenceSource()
        sig = data.get("signals", {})
        signals = EvidenceSignals.from_dict(sig) if isinstance(sig, dict) else EvidenceSignals()

        return cls(
            evidence_id=data["evidence_id"],
            type=data["type"],
            start=float(data["start"]),
            end=float(data["end"]),
            source=source,
            observation=data["observation"],
            signals=signals,
            confidence=float(data.get("confidence", 0.8)),
            reliability=data.get("reliability", "medium"),
            uncertainty=data.get("uncertainty", []),
            tags=data.get("tags", []),
        )


@dataclass
class EvidenceInventory:
    """Full standardized evidence repository."""
    version: str = "3.0.0"
    source_video: str = ""
    duration_seconds: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        for e in self.evidence:
            if e.evidence_id == evidence_id:
                return e
        return None

    def filter_by_type(self, evidence_type: str) -> list[EvidenceItem]:
        return [e for e in self.evidence if e.type == evidence_type]

    def filter_by_time_range(self, start: float, end: float) -> list[EvidenceItem]:
        return [e for e in self.evidence if not (e.end < start or e.start > end)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "duration_seconds": self.duration_seconds,
            "total_evidence_items": len(self.evidence),
            "evidence": [e.to_dict() for e in self.evidence],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceInventory:
        return cls(
            version=data.get("version", "3.0.0"),
            source_video=data.get("source_video", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            evidence=[EvidenceItem.from_dict(e) for e in data.get("evidence", [])],
            metadata=data.get("metadata", {}),
        )


# ─── Story Observation Models ─────────────────────────────────────────────────

@dataclass
class StoryObservation:
    """
    Narrative act or semantic structural observation.
    Must reference supporting evidence IDs.
    """
    story_observation_id: str
    type: str  # HOOK, CONTEXT, PROBLEM, BUILD_UP, EXAMPLE, CONTRAST, REALIZATION, PAYOFF, CTA, ENDING
    time_range: list[float]  # [start, end]
    confidence: float
    evidence_ids: list[str]
    summary: str
    signals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        upper_type = self.type.upper()
        if upper_type not in STORY_ACT_TYPES:
            raise ValueError(f"Invalid story observation type '{self.type}'. Must be one of {STORY_ACT_TYPES}")
        self.type = upper_type
        if len(self.time_range) != 2 or self.time_range[0] > self.time_range[1]:
            raise ValueError(f"Invalid time_range: {self.time_range}")
        self.confidence = max(0.0, min(1.0, round(float(self.confidence), 3)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_observation_id": self.story_observation_id,
            "type": self.type,
            "time_range": self.time_range,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "summary": self.summary,
            "signals": self.signals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoryObservation:
        return cls(
            story_observation_id=data["story_observation_id"],
            type=data["type"],
            time_range=[float(t) for t in data["time_range"]],
            confidence=float(data.get("confidence", 0.85)),
            evidence_ids=data.get("evidence_ids", []),
            summary=data.get("summary", ""),
            signals=data.get("signals", {}),
        )


@dataclass
class StoryObservationsInventory:
    """Complete narrative structure inventory."""
    version: str = "3.0.0"
    source_video: str = ""
    duration_seconds: float = 0.0
    story_observations: list[StoryObservation] = field(default_factory=list)
    narrative_health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "duration_seconds": self.duration_seconds,
            "total_story_observations": len(self.story_observations),
            "story_observations": [so.to_dict() for so in self.story_observations],
            "narrative_health": self.narrative_health,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoryObservationsInventory:
        return cls(
            version=data.get("version", "3.0.0"),
            source_video=data.get("source_video", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            story_observations=[
                StoryObservation.from_dict(so) for so in data.get("story_observations", [])
            ],
            narrative_health=data.get("narrative_health", {}),
        )


# ─── Edit Opportunity Models ──────────────────────────────────────────────────

@dataclass
class EditOpportunity:
    """
    Candidate opportunity identified by Agent 1 Observer for downstream evaluation.
    MUST reference evidence IDs.
    MUST NOT specify creative styling (no zoom ratios, no fonts, no colors, no easing).
    """
    opportunity_id: str
    type: str
    time_range: list[float]  # [start, end]
    evidence_ids: list[str]
    confidence: float
    priority: float
    description: str

    def __post_init__(self):
        if self.type not in OPPORTUNITY_TYPES:
            raise ValueError(f"Invalid opportunity type '{self.type}'. Supported types: {OPPORTUNITY_TYPES}")
        if len(self.time_range) != 2 or self.time_range[0] > self.time_range[1]:
            raise ValueError(f"Invalid time_range: {self.time_range}")
        self.confidence = max(0.0, min(1.0, round(float(self.confidence), 3)))
        self.priority = max(0.0, min(1.0, round(float(self.priority), 3)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "type": self.type,
            "time_range": self.time_range,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
            "priority": self.priority,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditOpportunity:
        return cls(
            opportunity_id=data["opportunity_id"],
            type=data["type"],
            time_range=[float(t) for t in data["time_range"]],
            evidence_ids=data.get("evidence_ids", []),
            confidence=float(data.get("confidence", 0.8)),
            priority=float(data.get("priority", 0.5)),
            description=data.get("description", ""),
        )


@dataclass
class OpportunityInventory:
    """Catalog of observed editing opportunities."""
    version: str = "3.0.0"
    source_video: str = ""
    duration_seconds: float = 0.0
    opportunities: list[EditOpportunity] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "duration_seconds": self.duration_seconds,
            "total_opportunities": len(self.opportunities),
            "opportunities": [o.to_dict() for o in self.opportunities],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpportunityInventory:
        return cls(
            version=data.get("version", "3.0.0"),
            source_video=data.get("source_video", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            opportunities=[EditOpportunity.from_dict(o) for o in data.get("opportunities", [])],
            metadata=data.get("metadata", {}),
        )
