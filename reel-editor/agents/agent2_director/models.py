"""
agents/agent2_director/models.py

Data Contracts for Agent 2: Story Editor (Phase 3)
─────────────────────────────────────────────────
Defines the typed data structures for:
  1. Edit Operations & Moment Classifications
  2. Edit Decision Data Model (conforming to Edit Decision Schema)
  3. Editorial Moment Analysis & Story Edit Plan
  4. Edit Decisions Document & Budget Utilization

Key Architectural Principles:
  - Agent 2 answers: "What should actually change in this reel?"
  - NO exact styling: no fonts, colors, zoom ratios (1.1x/1.25x), animation easings, or exact file assets.
  - DO NOT convert every opportunity into an edit (selective professional editing).
  - NO_EDIT is a first-class citizen.
  - Full evidence traceability: transcript → evidence → story observation → opportunity → decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ─── Supported Operations ─────────────────────────────────────────────────────

SUPPORTED_OPERATIONS = (
    "KEEP",
    "NO_EDIT",
    "REMOVE_FILLER",
    "REMOVE_DEAD_SPACE",
    "PACE_TRIM",
    "CUT",
    "SOFT_CUT",
    "PUNCH_IN",
    "EMPHASIS_TEXT",
    "HERO_TEXT",
    "EDITORIAL_CARD",
    "BROLL_SUPPORT",
    "BROLL_CONTEXT",
    "BROLL_CONTRAST",
    "SFX_WHOOSH",
    "SFX_POP",
    "SFX_IMPACT",
    "SFX_RISER",
    "MUSIC_ENTER",
    "MUSIC_EXIT",
    "AUDIO_DUCK",
)

MOMENT_CLASSIFICATIONS = (
    "PRIMARY",      # Key narrative anchor / core realization / hook / thesis
    "SECONDARY",    # Supporting context / transition / elaboration
    "PRESERVE",     # Natural delivery / emotional breathing room / unbroken rhetorical rhythm
)

PUNCH_IN_STRENGTHS = (
    "SUBTLE",       # Natural slight emphasis
    "MEDIUM",       # Noticeable framing shift for key argument
    "STRONG",       # High-impact framing jump for climactic thesis
)

DECISION_STATUSES = (
    "ACCEPTED",     # Enacted editorial intervention
    "REJECTED",     # Evaluated opportunity declined with reason
    "NO_EDIT",      # Explicit preservation of untouched span
)


# ─── Edit Decision Model ──────────────────────────────────────────────────────

@dataclass
class EditDecision:
    """
    Standardized Edit Decision emitted by Agent 2.
    Strictly answers WHAT should change and WHY, without creative styling commands.
    """
    decision_id: str
    operation: str
    status: str  # ACCEPTED | REJECTED | NO_EDIT
    start: float
    end: float
    source_evidence_ids: list[str] = field(default_factory=list)
    source_opportunity_ids: list[str] = field(default_factory=list)
    story_observation_ids: list[str] = field(default_factory=list)
    story_act: str = "CONTEXT"
    importance: float = 0.5
    confidence: float = 0.8
    reason: str = ""

    # Conceptual parameters (NO styling or file paths!)
    strength: Optional[str] = None          # for PUNCH_IN: SUBTLE | MEDIUM | STRONG
    text_content: Optional[str] = None      # for HERO_TEXT / EMPHASIS_TEXT: raw text words only
    concept_query: Optional[str] = None     # for B-roll: concept description, no file paths
    sound_cue: Optional[str] = None         # for SFX: accent | whoosh | impact | pop | riser
    trim_duration: Optional[float] = None   # for CUT / REMOVE_DEAD_SPACE / PACE_TRIM

    def __post_init__(self):
        if self.operation not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Invalid operation '{self.operation}'. Supported: {SUPPORTED_OPERATIONS}"
            )
        if self.status not in DECISION_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of {DECISION_STATUSES}"
            )
        if self.start < 0.0 or self.end < self.start:
            raise ValueError(f"Invalid time range [{self.start}, {self.end}]")

        if self.strength and self.strength not in PUNCH_IN_STRENGTHS:
            raise ValueError(
                f"Invalid strength '{self.strength}'. Must be one of {PUNCH_IN_STRENGTHS}"
            )

        self.start = round(float(self.start), 3)
        self.end = round(float(self.end), 3)
        self.importance = max(0.0, min(1.0, round(float(self.importance), 3)))
        self.confidence = max(0.0, min(1.0, round(float(self.confidence), 3)))
        if self.trim_duration is not None:
            self.trim_duration = round(float(self.trim_duration), 3)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "decision_id": self.decision_id,
            "operation": self.operation,
            "status": self.status,
            "start": self.start,
            "end": self.end,
            "source_evidence_ids": list(self.source_evidence_ids),
            "source_opportunity_ids": list(self.source_opportunity_ids),
            "story_observation_ids": list(self.story_observation_ids),
            "story_act": self.story_act,
            "importance": self.importance,
            "confidence": self.confidence,
            "reason": self.reason,
        }
        if self.strength is not None:
            d["strength"] = self.strength
        if self.text_content is not None:
            d["text_content"] = self.text_content
        if self.concept_query is not None:
            d["concept_query"] = self.concept_query
        if self.sound_cue is not None:
            d["sound_cue"] = self.sound_cue
        if self.trim_duration is not None:
            d["trim_duration"] = self.trim_duration
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditDecision:
        return cls(
            decision_id=data["decision_id"],
            operation=data["operation"],
            status=data.get("status", "ACCEPTED"),
            start=float(data["start"]),
            end=float(data["end"]),
            source_evidence_ids=list(data.get("source_evidence_ids", [])),
            source_opportunity_ids=list(data.get("source_opportunity_ids", [])),
            story_observation_ids=list(data.get("story_observation_ids", [])),
            story_act=data.get("story_act", "CONTEXT"),
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 0.8)),
            reason=data.get("reason", ""),
            strength=data.get("strength"),
            text_content=data.get("text_content"),
            concept_query=data.get("concept_query"),
            sound_cue=data.get("sound_cue"),
            trim_duration=float(data["trim_duration"]) if data.get("trim_duration") is not None else None,
        )


# ─── Editorial Moment & Narrative Plan ────────────────────────────────────────

@dataclass
class EditorialMoment:
    """Classified narrative moment with rhetorical priority."""
    moment_id: str
    story_act: str
    classification: str  # PRIMARY | SECONDARY | PRESERVE
    start: float
    end: float
    core_idea: str
    strongest_statement: str
    emotional_importance: float
    rhetorical_importance: float
    pacing_weakness: Optional[str] = None
    attention_peak: bool = False
    attention_drop: bool = False
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditorialMoment:
        return cls(**data)


@dataclass
class StoryEditPlan:
    """Strategic editorial overview produced by Agent 2."""
    version: str = "3.0.0"
    source_video: str = ""
    duration_seconds: float = 0.0
    pacing_preset: str = "balanced"
    narrative_strategy: str = ""
    moments: list[EditorialMoment] = field(default_factory=list)
    budget_limits: dict[str, Any] = field(default_factory=dict)
    decisions_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "duration_seconds": self.duration_seconds,
            "pacing_preset": self.pacing_preset,
            "narrative_strategy": self.narrative_strategy,
            "total_moments": len(self.moments),
            "moments": [m.to_dict() for m in self.moments],
            "budget_limits": self.budget_limits,
            "decisions_summary": self.decisions_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoryEditPlan:
        return cls(
            version=data.get("version", "3.0.0"),
            source_video=data.get("source_video", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            pacing_preset=data.get("pacing_preset", "balanced"),
            narrative_strategy=data.get("narrative_strategy", ""),
            moments=[EditorialMoment.from_dict(m) for m in data.get("moments", [])],
            budget_limits=data.get("budget_limits", {}),
            decisions_summary=data.get("decisions_summary", {}),
        )


# ─── Edit Decisions Document ──────────────────────────────────────────────────

@dataclass
class EditDecisionsDocument:
    """
    Complete collection of accepted, rejected, and NO_EDIT decisions.
    Conforms to Edit Decision Layer specification.
    """
    version: str = "3.0.0"
    source_video: str = ""
    duration_seconds: float = 0.0
    accepted_decisions: list[EditDecision] = field(default_factory=list)
    no_edit_decisions: list[EditDecision] = field(default_factory=list)
    rejected_decisions: list[EditDecision] = field(default_factory=list)
    budget_utilization: dict[str, Any] = field(default_factory=dict)
    traceability_matrix: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_decisions(self) -> list[EditDecision]:
        return self.accepted_decisions + self.no_edit_decisions + self.rejected_decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_video": self.source_video,
            "duration_seconds": self.duration_seconds,
            "total_accepted": len(self.accepted_decisions),
            "total_no_edit": len(self.no_edit_decisions),
            "total_rejected": len(self.rejected_decisions),
            "total_decisions": len(self.all_decisions),
            "accepted_decisions": [d.to_dict() for d in self.accepted_decisions],
            "no_edit_decisions": [d.to_dict() for d in self.no_edit_decisions],
            "rejected_decisions": [d.to_dict() for d in self.rejected_decisions],
            "budget_utilization": self.budget_utilization,
            "traceability_matrix": self.traceability_matrix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditDecisionsDocument:
        return cls(
            version=data.get("version", "3.0.0"),
            source_video=data.get("source_video", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            accepted_decisions=[
                EditDecision.from_dict(d) for d in data.get("accepted_decisions", [])
            ],
            no_edit_decisions=[
                EditDecision.from_dict(d) for d in data.get("no_edit_decisions", [])
            ],
            rejected_decisions=[
                EditDecision.from_dict(d) for d in data.get("rejected_decisions", [])
            ],
            budget_utilization=data.get("budget_utilization", {}),
            traceability_matrix=data.get("traceability_matrix", []),
        )
