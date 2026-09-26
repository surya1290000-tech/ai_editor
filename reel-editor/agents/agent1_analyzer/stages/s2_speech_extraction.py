"""
agents/agent1_analyzer/stages/s2_speech_extraction.py

Stage 2: Speech Extraction (Multilingual)
──────────────────────────────────────────
INPUT:  audio.wav (16kHz mono, from Stage 1)
OUTPUT: transcript_raw.json

Engine: core.asr.ASRManager (pluggable, multilingual)

What this stage produces:
  - Word-level timestamps with per-word language tags (en/te/hi/mixed)
  - Script detection (latin/telugu/devanagari)
  - Sentence boundaries with language_mix metadata
  - Silence / pause regions
  - Filler word flags per word
  - Low-confidence word flags with alternate hypotheses
  - Speech statistics including language distribution

What this stage does NOT do:
  - Audio feature analysis (energy, pitch) — that's Stage 3
  - Semantic understanding — that's Stage 6
  - Pause classification — that's Stage 5 (feature fusion)
  - Edit decisions — that's Agent 2

Design principles:
  - Agent 1 is an OBSERVER: transcribes honestly, never decides edits
  - Multilingual-first: Telugu, Hindi, English, code-switching are first-class
  - Never forces uncertain Telugu/Hindi into English
  - Low-confidence words are flagged, not silently discarded
  - Model is selected by hardware_profile, not hardcoded
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.logger import get_logger
from core.asr import ASRManager

logger = get_logger(__name__)


# ─── Public entry point ───────────────────────────────────────────────────────

def run(
    audio_path: Path,
    output_dir: Path,
    whisper_model: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    language_hint: Optional[str] = None,
) -> dict:
    """
    Run Stage 2: Transcribe audio with word-level timestamps and multilingual support.

    Args:
        audio_path:          Path to audio.wav from Stage 1.
        output_dir:          Directory to write transcript_raw.json.
        whisper_model:       Model size: small/medium/large-v3-turbo
        whisper_device:      "cpu" or "cuda"
        whisper_compute_type: "int8" (CPU) or "float16" (GPU)
        language_hint:       "en", "te", "hi", or None (auto-detect recommended)

    Returns:
        transcript dict (also saved as output_dir/transcript_raw.json)
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(
        f"Stage 2 | Initializing ASRManager "
        f"(model={whisper_model}, device={whisper_device}, "
        f"compute_type={whisper_compute_type})..."
    )

    # ── Initialize the ASR Manager ────────────────────────────────────────
    manager = ASRManager(
        model_size=whisper_model,
        device=whisper_device,
        compute_type=whisper_compute_type,
    )

    if not manager.is_available():
        raise ImportError(
            "faster-whisper is not installed. "
            "Run: pip install faster-whisper"
        )

    # ── Run transcription ─────────────────────────────────────────────────
    logger.info(
        f"Stage 2 | Transcribing '{audio_path.name}' "
        f"(language_hint={language_hint!r}, model={whisper_model})"
    )

    transcript = manager.transcribe(
        audio_path=audio_path,
        language_hint=language_hint,
    )

    # ── Convert to dict for JSON serialization ────────────────────────────
    transcript_dict = transcript.to_dict()

    # ── Add backward-compatible fields ────────────────────────────────────
    # These fields were expected by downstream stages in the old format.
    # We preserve them so Stage 3/5 don't break.
    transcript_dict["detected_language"] = transcript.primary_language
    transcript_dict["language_probability"] = transcript.language_confidence
    transcript_dict["language_hint_provided"] = language_hint

    # Convert ASRWord dicts to the flat format expected by downstream stages
    # (Stage 3 audio_features expects words with: id, word, start, end, duration,
    #  confidence, is_filler, low_confidence_flag, emphasis_score, energy_deviation, pitch_deviation)
    # Our new format already contains all these fields, so no conversion needed.

    # ── Save transcript ───────────────────────────────────────────────────
    from core.json_utils import save_json
    out_path = output_dir / "transcript_raw.json"
    save_json(transcript_dict, out_path)

    logger.info(f"Stage 2 | Complete. Transcript saved: {out_path.name}")
    logger.info(
        f"Stage 2 | Languages detected: {transcript.detected_languages}, "
        f"Code-switching: {transcript.speech_stats.get('code_switching_detected', False)}"
    )
    logger.info(f"Stage 2 | Full text preview: \"{transcript.full_text[:120]}...\"")

    return transcript_dict
