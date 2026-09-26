"""
core/asr/manager.py

Pluggable Multi-ASR Manager & Fusion Orchestrator
─────────────────────────────────────────────────
Orchestrates multilingual ASR across multiple specialized engines:
  1. Multilingual Whisper (English & code-switching alignment)
  2. AI4Bharat IndicConformer Telugu (Native Telugu phonetics & script)
  3. Evidence-based ASR Fusion (Conflict resolution, loanword validation)

Modes:
  - 'fused' (default): Dual-ASR execution on mixed speech + evidence fusion
  - 'whisper': Whisper-only execution
  - 'indic_conformer': IndicConformer-only execution
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.asr.base import ASRTranscript, ASRWord
from core.asr.whisper_engine import WhisperASREngine
from core.asr.indic_engine import IndicConformerEngine
from core.asr.fusion import ASRFusionEngine
from core.asr.segmenter import SpeechSegmenter, SpeechSegment
from core.logger import get_logger

logger = get_logger(__name__)


class ASRManager:
    """
    Pluggable Multi-ASR orchestrator supporting Whisper, IndicConformer, and Fusion.
    """

    def __init__(
        self,
        mode: str = "fused",
        whisper_model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[str] = None,
    ):
        self.mode = mode
        self.whisper_model_size = whisper_model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root

        # Initialize engines
        self.whisper_engine = WhisperASREngine(
            model_size=whisper_model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        self.indic_engine = IndicConformerEngine(
            device=device,
        )
        self.fusion_engine = ASRFusionEngine()
        self.segmenter = SpeechSegmenter()

        logger.info(
            f"ASRManager | Initialized in mode='{self.mode}' "
            f"(Whisper='{whisper_model_size}', IndicConformer='ai4bharat-indicconformer-te-onnx')"
        )

    @property
    def engine_name(self) -> str:
        if self.mode == "fused":
            return "multi_asr_fusion (Whisper + IndicConformer)"
        elif self.mode == "indic_conformer":
            return self.indic_engine.engine_name
        return self.whisper_engine.engine_name

    def is_available(self) -> bool:
        """Check availability of required engines."""
        if self.mode == "whisper":
            return self.whisper_engine.is_available()
        elif self.mode == "indic_conformer":
            return self.indic_engine.is_available()
        return self.whisper_engine.is_available() and self.indic_engine.is_available()

    def transcribe(
        self,
        audio_path: Path | str,
        language_hint: Optional[str] = None,
        force_dual_asr: bool = False,
    ) -> ASRTranscript:
        """
        Transcribe an audio file using the configured multi-ASR mode.

        Args:
            audio_path: Path to audio file (16kHz mono WAV preferred).
            language_hint: Optional hint ("en", "te", None).
            force_dual_asr: If True, forces both Whisper and IndicConformer regardless of language detection.
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(
            f"ASRManager | Transcribing '{audio_path.name}' in mode='{self.mode}' "
            f"(hint={language_hint!r}, force_dual={force_dual_asr})"
        )

        t0 = time.time()

        # ── Mode 1: Whisper-only ──────────────────────────────────────
        if self.mode == "whisper":
            return self.whisper_engine.transcribe(audio_path, language_hint)

        # ── Mode 2: IndicConformer-only ───────────────────────────────
        if self.mode == "indic_conformer":
            return self.indic_engine.transcribe(audio_path, language_hint)

        # ── Mode 3: Fused Dual-ASR Execution ──────────────────────────
        # Step A: Run Whisper
        logger.info("ASRManager | Running Whisper multilingual pass...")
        w_transcript = self.whisper_engine.transcribe(audio_path, language_hint)

        # Determine whether IndicConformer pass is needed
        needs_indic = force_dual_asr or (language_hint in ("te", "mixed"))
        if not needs_indic:
            has_te = "te" in w_transcript.detected_languages or w_transcript.primary_language == "te"
            has_indic_words = any(w.language == "te" for w in w_transcript.words)
            low_en_conf = (w_transcript.primary_language == "en" and w_transcript.language_confidence < 0.88)
            needs_indic = has_te or has_indic_words or low_en_conf

        if not needs_indic:
            logger.info(
                f"ASRManager | Confident pure English detected ({w_transcript.language_confidence:.2f}). "
                f"Skipping secondary IndicConformer pass."
            )
            return w_transcript

        # Step B: Run IndicConformer Telugu
        logger.info("ASRManager | Running AI4Bharat IndicConformer Telugu pass...")
        try:
            i_transcript = self.indic_engine.transcribe(audio_path, language_hint)
        except Exception as e:
            logger.error(f"ASRManager | IndicConformer pass failed: {e}. Falling back to Whisper.")
            return w_transcript

        # Step C: Evidence-Based Fusion
        logger.info("ASRManager | Fusing hypotheses from Whisper & IndicConformer...")
        fused_transcript = self.fusion_engine.fuse(w_transcript, i_transcript)

        total_time = round(time.time() - t0, 2)
        fused_transcript.transcription_time_seconds = total_time

        logger.info(
            f"ASRManager | Fused transcription complete in {total_time}s. "
            f"Total words: {len(fused_transcript.words)}, Agreement: {fused_transcript.speech_stats.get('agreement_rate', 1.0):.2f}"
        )
        return fused_transcript

    def transcribe_to_dict(
        self,
        audio_path: Path | str,
        language_hint: Optional[str] = None,
        force_dual_asr: bool = False,
    ) -> Dict[str, Any]:
        """Transcribe and return JSON-serializable dictionary."""
        transcript = self.transcribe(audio_path, language_hint, force_dual_asr)
        return transcript.to_dict()
