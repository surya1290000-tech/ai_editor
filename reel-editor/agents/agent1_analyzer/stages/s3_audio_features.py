"""
agents/agent1_analyzer/stages/s3_audio_features.py

Stage 3: Audio Signal Analysis
──────────────────────────────
INPUT:  audio.wav (16kHz mono, from Stage 1)
        transcript_raw.json (word list, from Stage 2)
OUTPUT: audio_features.json

Tools: librosa

What this stage measures (all deterministic — no LLM):
  - RMS energy envelope (50ms hops) → speaker baseline energy
  - Per-word energy deviation from baseline
  - Pitch (F0) via pyin → speaker baseline pitch
  - Per-word pitch deviation from baseline
  - Per-sentence speaking rate (WPM) + deviation from speaker average
  - Per-word EMPHASIS SCORE (multi-signal formula)
  - Silence region energy context (preceding/following energy)
  - Audio quality: noise floor, clipping detection

What this stage does NOT do:
  - Pause classification (needs semantic context — Stage 5)
  - Story understanding (Stage 6)
  - Any LLM calls

Design notes:
  - 50ms hop = ~20 measurements per second. Fine-grained enough for word-level alignment.
  - Speaker baseline is computed as the MEDIAN, not mean, so outliers (shouts, whispers)
    don't skew the reference.
  - Emphasis formula weights are loaded from editorial_rules.yaml (tunable).
  - All per-word signals are written back into the transcript's word list so
    Stage 5 (feature fusion) can build a unified per-sentence timeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.logger import get_logger

logger = get_logger(__name__)

# Hop length for short-time analysis (50ms at 16kHz = 800 samples)
_HOP_LENGTH = 800
_SAMPLE_RATE = 16000
_HOP_MS = 1000 * _HOP_LENGTH / _SAMPLE_RATE  # = 50ms


def _load_rules() -> dict:
    """Load editorial_rules.yaml for tunable thresholds."""
    import yaml
    rules_path = Path(__file__).parent.parent.parent.parent / "config" / "editorial_rules.yaml"
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Public entry point ───────────────────────────────────────────────────────

def run(
    audio_path: Path,
    transcript: dict,
    output_dir: Path,
) -> dict:
    """
    Run Stage 3: Extract audio features and compute per-word emphasis scores.

    Args:
        audio_path:  Path to audio.wav from Stage 1.
        transcript:  Dict from Stage 2 (transcript_raw.json).
        output_dir:  Directory to write audio_features.json.

    Returns:
        audio_features dict (also saved as output_dir/audio_features.json)
    """
    import librosa

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    logger.info("Stage 3 | Loading audio...")
    y, sr = librosa.load(str(audio_path), sr=_SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    logger.info(f"Stage 3 | Audio loaded: {duration:.1f}s at {sr}Hz")

    rules = _load_rules()
    emphasis_weights = rules.get("emphasis_weights", {
        "energy_deviation": 0.35,
        "pitch_deviation": 0.25,
        "rate_slowdown": 0.20,
        "pause_proximity": 0.20,
    })

    # ── 1. RMS Energy ─────────────────────────────────────────────
    logger.info("Stage 3 | Computing RMS energy envelope...")
    rms = librosa.feature.rms(y=y, hop_length=_HOP_LENGTH)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=_HOP_LENGTH
    )

    # Speaker energy baseline: median of frames with speech
    words = transcript.get("words", [])
    rms_at_speech = _get_signal_at_speech(rms, rms_times, words)
    baseline_energy = float(np.median(rms_at_speech)) if len(rms_at_speech) > 0 else float(np.median(rms))
    logger.info(f"Stage 3 | Speaker energy baseline: {baseline_energy:.4f} RMS")

    # ── 2. Pitch (F0) ─────────────────────────────────────────────
    logger.info("Stage 3 | Computing pitch (F0) via pyin... (this takes ~30s on CPU)")
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz — covers all human voices
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — above any speech pitch
            sr=sr,
            hop_length=_HOP_LENGTH,
            fill_na=0.0,
        )
        pitch_times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=_HOP_LENGTH
        )
        # Speaker pitch baseline: median of voiced frames only
        voiced_f0 = f0[voiced_flag & (f0 > 0)]
        baseline_pitch = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        logger.info(f"Stage 3 | Speaker pitch baseline: {baseline_pitch:.1f} Hz")
        pitch_available = True
    except Exception as e:
        logger.warning(f"Stage 3 | Pitch computation failed: {e}. Emphasis will use energy+rate only.")
        f0 = np.zeros(len(rms))
        pitch_times = rms_times
        baseline_pitch = 0.0
        voiced_flag = np.zeros(len(rms), dtype=bool)
        pitch_available = False

    # ── 3. Per-word energy and pitch deviation ────────────────────
    logger.info("Stage 3 | Computing per-word signals...")
    words_with_signals = _compute_word_signals(
        words, y, sr, rms, rms_times, f0, pitch_times,
        baseline_energy, baseline_pitch, pitch_available
    )

    # ── 4. Per-sentence speaking rate ─────────────────────────────
    sentences = transcript.get("sentences", [])
    sentences_with_rate = _compute_sentence_rates(sentences, words_with_signals)
    speaker_baseline_wpm = _compute_wpm_baseline(sentences_with_rate)
    logger.info(f"Stage 3 | Speaker baseline WPM: {speaker_baseline_wpm:.0f}")

    # Add rate deviation to each sentence
    for s in sentences_with_rate:
        if s["words_per_minute"] and speaker_baseline_wpm > 0:
            s["rate_deviation"] = round(
                (s["words_per_minute"] - speaker_baseline_wpm) / speaker_baseline_wpm, 3
            )

    # ── 5. Emphasis score per word ────────────────────────────────
    logger.info("Stage 3 | Computing multi-signal emphasis scores...")
    words_with_emphasis = _compute_emphasis_scores(
        words_with_signals, sentences_with_rate, speaker_baseline_wpm, emphasis_weights
    )

    # ── 6. Silence region energy context ─────────────────────────
    silence_regions = transcript.get("silence_regions", [])
    silence_with_energy = _annotate_silence_energy(silence_regions, rms, rms_times, baseline_energy)

    # ── 7. Audio quality metrics ──────────────────────────────────
    quality = _compute_quality_metrics(y, sr, rms, words)

    # ── 8. Full energy/pitch envelopes (sampled for storage) ──────
    # Store at 4-sample intervals to reduce file size while preserving detail
    envelope_sample = 4
    energy_envelope = [
        {"t": round(float(t), 3), "rms": round(float(v), 5)}
        for t, v in zip(rms_times[::envelope_sample], rms[::envelope_sample])
    ]

    audio_features = {
        "baseline_energy": round(baseline_energy, 5),
        "baseline_pitch_hz": round(baseline_pitch, 2),
        "speaker_baseline_wpm": round(speaker_baseline_wpm, 1),
        "pitch_available": pitch_available,
        "emphasis_weights_used": emphasis_weights,
        "words": words_with_emphasis,       # enriched word list
        "sentences": sentences_with_rate,   # enriched sentence list
        "silence_regions": silence_with_energy,
        "quality": quality,
        "energy_envelope": energy_envelope,
    }

    from core.json_utils import save_json
    out_path = output_dir / "audio_features.json"
    save_json(audio_features, out_path)

    logger.info(f"Stage 3 | Complete. Audio features saved: {out_path.name}")
    _log_emphasis_summary(words_with_emphasis)

    return audio_features


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_signal_at_speech(signal: np.ndarray, times: np.ndarray, words: list) -> np.ndarray:
    """Return signal values that correspond to speech regions (word start/end)."""
    speech_values = []
    for w in words:
        mask = (times >= w["start"]) & (times <= w["end"])
        speech_values.extend(signal[mask].tolist())
    return np.array(speech_values) if speech_values else np.array([])


def _mean_signal_in_window(signal: np.ndarray, times: np.ndarray, t_start: float, t_end: float) -> float:
    """Mean of signal values within a time window."""
    mask = (times >= t_start) & (times <= t_end)
    values = signal[mask]
    return float(np.mean(values)) if len(values) > 0 else 0.0


def _compute_word_signals(
    words: list,
    y: np.ndarray,
    sr: int,
    rms: np.ndarray,
    rms_times: np.ndarray,
    f0: np.ndarray,
    pitch_times: np.ndarray,
    baseline_energy: float,
    baseline_pitch: float,
    pitch_available: bool,
) -> list:
    """
    Compute energy deviation and pitch deviation for each word.
    Returns enriched word list.
    """
    enriched = []
    for w in words:
        w = dict(w)  # don't mutate original

        # Mean energy during this word
        word_energy = _mean_signal_in_window(rms, rms_times, w["start"], w["end"])
        # Deviation relative to baseline: >0 = louder, <0 = quieter
        if baseline_energy > 0:
            energy_dev = (word_energy - baseline_energy) / baseline_energy
        else:
            energy_dev = 0.0

        # Mean pitch during this word (only voiced frames)
        pitch_dev = 0.0
        if pitch_available and baseline_pitch > 0:
            word_f0 = _mean_signal_in_window(f0, pitch_times, w["start"], w["end"])
            if word_f0 > 0:
                pitch_dev = (word_f0 - baseline_pitch) / baseline_pitch

        w["energy_deviation"] = round(energy_dev, 4)
        w["pitch_deviation"] = round(pitch_dev, 4)
        enriched.append(w)

    return enriched


def _compute_sentence_rates(sentences: list, words: list) -> list:
    """
    Compute WPM for each sentence using actual word durations.
    """
    # Build word lookup by ID for fast access
    word_by_id = {w["id"]: w for w in words}

    enriched = []
    for s in sentences:
        s = dict(s)
        dur = s.get("duration", 0)
        wc = s.get("word_count", 0)
        if dur > 0.3 and wc >= 2:
            wpm = round((wc / dur) * 60, 1)
        else:
            wpm = None
        s["words_per_minute"] = wpm
        enriched.append(s)
    return enriched


def _compute_wpm_baseline(sentences: list) -> float:
    """Median WPM across sentences with reliable duration."""
    rates = [s["words_per_minute"] for s in sentences if s.get("words_per_minute")]
    if not rates:
        return 130.0  # reasonable default
    return float(np.median(rates))


def _compute_emphasis_scores(
    words: list,
    sentences: list,
    baseline_wpm: float,
    weights: dict,
) -> list:
    """
    Compute a multi-signal emphasis score (0.0–1.0) per word.

    Formula:
      emphasis = clamp(
        w1 * energy_norm +
        w2 * pitch_norm +
        w3 * rate_slowdown_norm +
        w4 * pause_proximity_score
      , 0, 1)

    Normalisation:
      - energy/pitch deviation are normalised to 0–1 using clamp(dev / 2.0 + 0.5, 0, 1)
        (deviation of +2.0 = fully loud, deviation of -2.0 = fully quiet)
        Symmetric around 0.5 (baseline = no emphasis)
        We use |deviation| since both very loud AND very quiet can indicate emphasis
      - Rate slowdown: slower = more emphasis. Normalised by sentence rate vs baseline.
      - Pause proximity: silence within 300ms before or after word.
    """
    w1 = weights.get("energy_deviation", 0.35)
    w2 = weights.get("pitch_deviation", 0.25)
    w3 = weights.get("rate_slowdown", 0.20)
    w4 = weights.get("pause_proximity", 0.20)

    # Build sentence rate lookup by sentence word_ids
    word_sentence_rate: dict[int, float] = {}
    for s in sentences:
        rate = s.get("words_per_minute") or baseline_wpm
        for wid in s.get("word_ids", []):
            word_sentence_rate[wid] = rate

    enriched = []
    for i, w in enumerate(words):
        w = dict(w)

        # Energy component: use absolute deviation, louder OR quieter than baseline
        energy_dev = abs(w.get("energy_deviation") or 0.0)
        energy_norm = min(energy_dev / 1.5, 1.0)  # 1.5x deviation = full score

        # Pitch component: absolute deviation
        pitch_dev = abs(w.get("pitch_deviation") or 0.0)
        pitch_norm = min(pitch_dev / 0.5, 1.0)  # 50% pitch shift = full score

        # Rate component: compare sentence WPM to baseline
        sentence_wpm = word_sentence_rate.get(w["id"], baseline_wpm)
        if baseline_wpm > 0 and sentence_wpm > 0:
            # Slower sentence = higher emphasis score (slower = deliberate)
            rate_ratio = baseline_wpm / sentence_wpm  # >1 = slower than baseline
            rate_norm = min(max(rate_ratio - 1.0, 0.0) * 3.0, 1.0)
        else:
            rate_norm = 0.0

        # Pause proximity: is there a silence within 300ms of this word?
        pause_score = 0.0
        if i > 0:
            gap_before = w["start"] - words[i - 1]["end"]
            if gap_before >= 0.15:   # 150ms before
                pause_score = min(gap_before / 0.5, 1.0)
        if i < len(words) - 1:
            gap_after = words[i + 1]["start"] - w["end"]
            if gap_after >= 0.15:
                pause_score = max(pause_score, min(gap_after / 0.5, 1.0))

        emphasis = (
            w1 * energy_norm +
            w2 * pitch_norm +
            w3 * rate_norm +
            w4 * pause_score
        )
        emphasis = round(min(max(emphasis, 0.0), 1.0), 3)

        w["emphasis_score"] = emphasis
        w["_emphasis_components"] = {   # kept for debugging/auditability
            "energy_norm": round(energy_norm, 3),
            "pitch_norm": round(pitch_norm, 3),
            "rate_norm": round(rate_norm, 3),
            "pause_score": round(pause_score, 3),
        }
        enriched.append(w)

    return enriched


def _annotate_silence_energy(
    silence_regions: list,
    rms: np.ndarray,
    rms_times: np.ndarray,
    baseline_energy: float,
) -> list:
    """Add preceding/following energy levels to each silence region."""
    enriched = []
    for s in silence_regions:
        s = dict(s)
        look_window = 0.3  # 300ms before/after silence

        pre_energy = _mean_signal_in_window(
            rms, rms_times,
            max(0, s["start"] - look_window), s["start"]
        )
        post_energy = _mean_signal_in_window(
            rms, rms_times,
            s["end"], s["end"] + look_window
        )

        if baseline_energy > 0:
            s["preceding_energy"] = round(pre_energy, 5)
            s["following_energy"] = round(post_energy, 5)
            s["preceding_energy_normalized"] = round(pre_energy / baseline_energy, 3)
            s["following_energy_normalized"] = round(post_energy / baseline_energy, 3)
        else:
            s["preceding_energy"] = round(pre_energy, 5)
            s["following_energy"] = round(post_energy, 5)
            s["preceding_energy_normalized"] = 1.0
            s["following_energy_normalized"] = 1.0

        enriched.append(s)
    return enriched


def _compute_quality_metrics(
    y: np.ndarray, sr: int, rms: np.ndarray, words: list
) -> dict:
    """
    Compute audio quality indicators.
    All thresholds from editorial_rules.yaml.quality_warnings.audio.
    """
    # Clipping: samples very close to ±1.0
    clip_threshold = 0.98
    clipping_detected = bool(np.any(np.abs(y) > clip_threshold))
    clipping_frames = int(np.sum(np.abs(y) > clip_threshold))

    # Noise floor: estimate from quiet moments (lowest 10th percentile of RMS)
    noise_floor_rms = float(np.percentile(rms, 10))
    noise_floor_db = 20 * np.log10(max(noise_floor_rms, 1e-10))

    # Background noise level classification
    if noise_floor_db < -55:
        noise_level = "low"        # very clean
    elif noise_floor_db < -45:
        noise_level = "moderate"   # acceptable but present
    else:
        noise_level = "high"       # may affect transcription

    # Speech ratio (speech duration / total duration)
    total_speech = sum(w["end"] - w["start"] for w in words)
    total_duration = len(y) / sr
    speech_ratio = round(total_speech / max(total_duration, 1), 3)

    quality_warnings = []
    if clipping_detected:
        quality_warnings.append(
            f"Audio clipping detected ({clipping_frames} samples above {clip_threshold}). "
            "May cause distortion in transcription."
        )
    if noise_level == "high":
        quality_warnings.append(
            f"High background noise level ({noise_floor_db:.1f} dB). "
            "Consider noise reduction before analysis."
        )
    if speech_ratio < 0.40:
        quality_warnings.append(
            f"Low speech ratio ({speech_ratio:.0%}). Less than 40% of the video contains speech."
        )

    return {
        "clipping_detected": clipping_detected,
        "clipping_sample_count": clipping_frames,
        "noise_floor_db": round(noise_floor_db, 1),
        "noise_level": noise_level,
        "speech_ratio": speech_ratio,
        "quality_warnings": quality_warnings,
    }


def _log_emphasis_summary(words: list) -> None:
    """Log the top-5 emphasis words for quick inspection."""
    scored = sorted(words, key=lambda w: w.get("emphasis_score", 0), reverse=True)
    top5 = scored[:5]
    logger.info("Stage 3 | Top-5 emphasis words:")
    for w in top5:
        comps = w.get("_emphasis_components", {})
        logger.info(
            f"  [{w['emphasis_score']:.3f}] \"{w['word']}\" "
            f"@ {w['start']:.1f}s  "
            f"(energy={comps.get('energy_norm',0):.2f}, "
            f"pitch={comps.get('pitch_norm',0):.2f}, "
            f"rate={comps.get('rate_norm',0):.2f}, "
            f"pause={comps.get('pause_score',0):.2f})"
        )
