"""
core/asr/segmenter.py

Audio VAD Segmentation & Language / Code-Switching Classifier
─────────────────────────────────────────────────────────────
Partitions long audio files into coherent speech segments and classifies
each segment as:
  - 'en'      (pure English)
  - 'te'      (pure Telugu)
  - 'mixed'   (code-switched / mixed language)
  - 'unknown' (ambiguous or low confidence)

Segment Routing Rules:
  - 'en':      routed to Whisper
  - 'te':      routed to IndicConformer Telugu
  - 'mixed':   routed to Dual-ASR (Whisper + IndicConformer)
  - 'unknown': routed to Dual-ASR with uncertainty tracking
"""

from __future__ import annotations

import re
import math
import wave
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpeechSegment:
    """A coherent window of speech with language classification."""
    segment_id: int
    start: float
    end: float
    duration: float
    language_label: str       # "en", "te", "mixed", "unknown"
    confidence: float = 0.85
    audio_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "language_label": self.language_label,
            "confidence": round(self.confidence, 3),
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "metadata": self.metadata,
        }


class SpeechSegmenter:
    """
    Partitions audio using VAD boundaries and performs segment-level language routing.
    """

    def __init__(
        self,
        min_segment_sec: float = 1.5,
        max_segment_sec: float = 12.0,
        silence_threshold_sec: float = 0.40,
    ):
        self.min_segment_sec = min_segment_sec
        self.max_segment_sec = max_segment_sec
        self.silence_threshold_sec = silence_threshold_sec

    def segment_audio(
        self,
        audio_path: Path | str,
        output_dir: Optional[Path] = None,
        whisper_probe_func: Optional[Any] = None,
    ) -> List[SpeechSegment]:
        """
        Segments an audio file into speech windows and classifies language.

        Args:
            audio_path: Path to 16kHz mono WAV file.
            output_dir: Optional directory to save sliced segment WAV files.
            whisper_probe_func: Optional callback (slice_path -> language_info) for classification.
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Read audio data
        with wave.open(str(audio_path), "rb") as wf:
            samplerate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
            total_duration = n_frames / samplerate

        audio_samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Simple energy-based VAD / pause detection
        frame_len = int(samplerate * 0.05)  # 50ms frames
        n_steps = len(audio_samples) // frame_len
        energies = []

        for step in range(n_steps):
            frame = audio_samples[step * frame_len : (step + 1) * frame_len]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-9)
            energies.append(rms)

        # Detect silence regions (threshold at 15% of median speech energy)
        speech_rms = [e for e in energies if e > 0.01]
        threshold = np.percentile(speech_rms, 20) if speech_rms else 0.005

        silence_frames = [e < threshold for e in energies]

        # Find pause split points
        split_times = [0.0]
        cur_silence_duration = 0.0
        frame_duration = 0.05

        for idx, is_sil in enumerate(silence_frames):
            t = idx * frame_duration
            if is_sil:
                cur_silence_duration += frame_duration
            else:
                if cur_silence_duration >= self.silence_threshold_sec:
                    pause_mid = t - (cur_silence_duration / 2.0)
                    last_split = split_times[-1]
                    if (pause_mid - last_split) >= self.min_segment_sec:
                        split_times.append(round(pause_mid, 3))
                cur_silence_duration = 0.0

        # Ensure final split
        if total_duration - split_times[-1] >= self.min_segment_sec:
            split_times.append(round(total_duration, 3))
        else:
            split_times[-1] = round(total_duration, 3)

        # Build segments
        raw_intervals: List[Tuple[float, float]] = []
        for i in range(len(split_times) - 1):
            st = split_times[i]
            en = split_times[i + 1]

            # If segment is too long, subdivide
            if en - st > self.max_segment_sec:
                mid = (st + en) / 2.0
                raw_intervals.append((st, mid))
                raw_intervals.append((mid, en))
            else:
                raw_intervals.append((st, en))

        # Classify each segment
        segments: List[SpeechSegment] = []
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (st, en) in enumerate(raw_intervals):
            dur = round(en - st, 3)
            slice_path = None

            if output_dir:
                slice_path = output_dir / f"seg_{idx:03d}_{st:.2f}_{en:.2f}.wav"
                self._save_audio_slice(audio_samples, samplerate, st, en, slice_path)

            # Determine language label
            label = "mixed"  # Default for multilingual Telugu content
            conf = 0.85

            if whisper_probe_func and slice_path:
                try:
                    probe = whisper_probe_func(slice_path)
                    lang = probe.get("language")
                    prob = probe.get("probability", 0.5)
                    if prob > 0.85 and lang == "en":
                        label = "en"
                        conf = prob
                    elif prob > 0.80 and lang in ("te", "hi"):
                        label = "te"
                        conf = prob
                    elif 0.30 <= prob <= 0.85:
                        label = "mixed"
                        conf = 0.80
                    else:
                        label = "unknown"
                        conf = prob
                except Exception as e:
                    logger.warning(f"SpeechSegmenter | Probe failed for segment {idx}: {e}")
                    label = "mixed"

            segments.append(
                SpeechSegment(
                    segment_id=idx,
                    start=st,
                    end=en,
                    duration=dur,
                    language_label=label,
                    confidence=conf,
                    audio_path=slice_path,
                )
            )

        logger.info(f"SpeechSegmenter | Segmented audio into {len(segments)} speech windows.")
        return segments

    @staticmethod
    def _save_audio_slice(
        audio_samples: np.ndarray,
        samplerate: int,
        start_sec: float,
        end_sec: float,
        out_file: Path,
    ):
        """Extracts and writes a segment slice to disk."""
        start_sample = max(0, int(start_sec * samplerate))
        end_sample = min(len(audio_samples), int(end_sec * samplerate))
        slice_data = (audio_samples[start_sample:end_sample] * 32768.0).astype(np.int16)

        with wave.open(str(out_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(slice_data.tobytes())
