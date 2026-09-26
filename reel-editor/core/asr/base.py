"""
core/asr/base.py

Abstract ASR Engine Interface & Data Contracts
───────────────────────────────────────────────
Defines the typed data structures for multilingual ASR output
and the abstract base class that all ASR engines must implement.

Design principles:
  - Every word carries its own language tag, confidence, and timing
  - Code-switching is first-class: a sentence can contain mixed languages
  - Uncertain Telugu is never forced into English
  - alternate_hypotheses are provided when confidence is low
  - The contract is serializable to JSON for inspection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import re


# ─── Unicode Range Detection ─────────────────────────────────────────────────

# Telugu Unicode block: U+0C00 – U+0C7F
_TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')
# Devanagari Unicode block: U+0900 – U+097F
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def detect_script(text: str) -> str:
    """Detect the script of a text string based on Unicode code points."""
    if _TELUGU_RE.search(text):
        return "telugu"
    if _DEVANAGARI_RE.search(text):
        return "devanagari"
    return "latin"


# ─── Data Contracts ──────────────────────────────────────────────────────────

@dataclass
class ASRCandidate:
    """A candidate hypothesis from a specific ASR engine for auditability."""
    engine: str                       # e.g. "whisper", "indic_conformer"
    text: str
    confidence: float
    language: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ASRAlternative:
    """An alternate transcription hypothesis for a low-confidence word."""
    word: str
    confidence: float
    language: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ASRWord:
    """
    A single word with full multilingual metadata.

    Every word produced by the ASR subsystem carries:
    - Precise timing (start/end in seconds)
    - Language tag (ISO 639-1: en, te, hi, mixed)
    - Script detection (latin, telugu, devanagari)
    - Confidence score from the ASR engine
    - Filler word flag
    - Full candidate audit trail (competing engines & selection reason)
    - Spoken text vs Romanized text vs Native script text
    """
    id: int
    word: str
    start: float
    end: float
    duration: float
    confidence: float
    language: str = "en"              # ISO 639-1: "en", "te", "hi", "mixed"
    script: str = "latin"             # "latin", "telugu", "devanagari"
    spoken_text: str = ""             # Verbatim spoken token
    romanized: Optional[str] = None   # Romanized form when script != latin (legacy compatibility)
    romanized_text: Optional[str] = None  # Explicit Romanized representation
    native_script_text: Optional[str] = None  # Native script (e.g. Telugu Unicode)
    source_engine: str = "whisper"    # "whisper", "indic_conformer", "fused"
    candidates: List[ASRCandidate] = field(default_factory=list)
    agreement_score: float = 1.0
    selection_reason: str = ""
    is_filler: bool = False
    low_confidence_flag: bool = False
    alternate_hypotheses: List[ASRAlternative] = field(default_factory=list)
    # Acoustic features (populated by Stage 3, not ASR)
    emphasis_score: Optional[float] = None
    energy_deviation: Optional[float] = None
    pitch_deviation: Optional[float] = None

    def __post_init__(self):
        if not self.spoken_text:
            self.spoken_text = self.word
        if self.romanized and not self.romanized_text:
            self.romanized_text = self.romanized
        elif self.romanized_text and not self.romanized:
            self.romanized = self.romanized_text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alternate_hypotheses"] = [a.to_dict() for a in self.alternate_hypotheses]
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ASRWord:
        d = dict(data)
        alts = d.pop("alternate_hypotheses", [])
        cands = d.pop("candidates", [])
        word = cls(**d)
        word.alternate_hypotheses = [
            ASRAlternative(**a) if isinstance(a, dict) else a
            for a in alts
        ]
        word.candidates = [
            ASRCandidate(**c) if isinstance(c, dict) else c
            for c in cands
        ]
        return word


@dataclass
class ASRSentence:
    """
    A sentence boundary derived from ASR segments and punctuation.
    Carries language mix information for code-switched sentences.
    """
    id: int
    text: str
    start: float
    end: float
    duration: float
    word_count: int
    word_ids: List[int]
    language_mix: List[str]           # e.g. ["te", "en"] for code-switched
    filler_count: int = 0
    has_filler: bool = False
    low_confidence_flag: bool = False
    # Populated by Stage 5 (Feature Fusion)
    speaking_rate_wpm: Optional[float] = None
    rate_deviation: Optional[float] = None
    avg_energy_deviation: Optional[float] = None
    peak_energy_deviation: Optional[float] = None
    top_emphasis_word: Optional[Dict[str, Any]] = None
    pause_before: Optional[Dict[str, Any]] = None
    pause_after: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ASRSentence:
        return cls(**data)


@dataclass
class ASRTranscript:
    """
    Complete multilingual transcript output from the ASR subsystem.

    This is the authoritative transcript contract between the ASR layer
    and all downstream consumers (Feature Fusion, Story Analysis, etc.).
    """
    full_text: str
    detected_languages: List[str]     # All languages found, e.g. ["te", "en"]
    primary_language: str             # Dominant language
    language_confidence: float        # Overall language detection confidence
    whisper_model: str                # Model used for transcription
    words: List[ASRWord] = field(default_factory=list)
    sentences: List[ASRSentence] = field(default_factory=list)
    silence_regions: List[Dict[str, Any]] = field(default_factory=list)
    speech_stats: Dict[str, Any] = field(default_factory=dict)
    # ASR engine metadata
    engine_name: str = "whisper"
    compute_type: str = "int8"
    device: str = "cpu"
    transcription_time_seconds: float = 0.0
    peak_ram_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "full_text": self.full_text,
            "detected_languages": self.detected_languages,
            "primary_language": self.primary_language,
            "language_confidence": self.language_confidence,
            "whisper_model": self.whisper_model,
            "words": [w.to_dict() for w in self.words],
            "sentences": [s.to_dict() for s in self.sentences],
            "silence_regions": self.silence_regions,
            "speech_stats": self.speech_stats,
            "engine_name": self.engine_name,
            "compute_type": self.compute_type,
            "device": self.device,
            "transcription_time_seconds": self.transcription_time_seconds,
            "peak_ram_mb": self.peak_ram_mb,
        }


# ─── Filler Word Detection ──────────────────────────────────────────────────

# English fillers
_ENGLISH_FILLERS = {
    "um", "uh", "hmm", "ah", "like", "basically", "literally",
    "honestly", "actually", "right", "okay", "ok", "you know",
    "i mean", "kind of", "sort of", "you see", "well", "so",
    "anyway",
}

# Telugu/Hindi fillers (Romanized)
_INDIC_FILLERS = {
    "matlab", "woh", "bas", "toh", "na", "yaar", "acha",
    "arrey", "matlab ki", "iska matlab", "ante", "adi",
    "ala", "inka",
}

ALL_FILLERS = _ENGLISH_FILLERS | _INDIC_FILLERS


def is_filler_word(word_text: str) -> bool:
    """Check if a word/phrase is a filler word (case-insensitive)."""
    cleaned = word_text.lower().strip().strip(".,!?;:")
    return cleaned in ALL_FILLERS


# ─── Abstract Engine ─────────────────────────────────────────────────────────

class BaseASREngine(ABC):
    """
    Abstract base class for pluggable ASR engines.

    All ASR engines must implement:
    - transcribe(): Takes an audio file path and returns an ASRTranscript
    - is_available(): Checks if the engine's dependencies are installed

    The engine should NOT make editorial decisions.
    It should produce honest, evidence-quality transcription data.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language_hint: Optional[str] = None,
    ) -> ASRTranscript:
        """
        Transcribe an audio file and return a complete ASRTranscript.

        Args:
            audio_path: Path to audio file (WAV, 16kHz mono preferred).
            language_hint: Optional ISO 639-1 hint ("en", "te", "hi", None for auto-detect).

        Returns:
            ASRTranscript with word-level timestamps, language tags, and confidence scores.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine's dependencies are installed and loadable."""
        ...

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable name of this ASR engine."""
        ...
