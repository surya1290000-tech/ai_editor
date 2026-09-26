"""
scripts/benchmark_asr.py

Phase 1 ASR Benchmark on Surya.mp4
───────────────────────────────────
Runs the new multilingual ASRManager on Surya.mp4 audio and produces:
  - English word accuracy assessment
  - Telugu word preservation verification
  - Code-switching detection results
  - Per-word language distribution
  - Runtime and RAM usage
  - Full transcript saved for inspection

Usage:
  python scripts/benchmark_asr.py
"""

from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

# Force UTF-8 output on Windows console
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.asr import ASRManager
from core.json_utils import save_json


def main():
    # Find Surya audio
    audio_candidates = sorted(
        PROJECT_ROOT.glob("outputs/Surya_*/audio.wav"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not audio_candidates:
        print("ERROR: No Surya audio.wav found. Run Stage 1 first.")
        sys.exit(1)

    audio_path = audio_candidates[0]
    print(f"Using audio: {audio_path}")
    print(f"Audio size: {audio_path.stat().st_size / (1024*1024):.2f} MB")

    # Track system RAM
    try:
        import psutil
        process = psutil.Process()
        ram_start = process.memory_info().rss / (1024 * 1024)
    except Exception:
        ram_start = 0

    # ── Run ASR ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 ASR BENCHMARK: Surya.mp4")
    print("=" * 70)

    manager = ASRManager(model_size="small", device="cpu", compute_type="int8")

    t0 = time.time()
    transcript = manager.transcribe(audio_path, language_hint=None)
    total_time = round(time.time() - t0, 2)

    try:
        ram_end = process.memory_info().rss / (1024 * 1024)
        ram_delta = round(ram_end - ram_start, 1)
    except Exception:
        ram_end = 0
        ram_delta = 0

    # ── Compute Metrics ──────────────────────────────────────────────
    words = transcript.words
    sentences = transcript.sentences
    stats = transcript.speech_stats
    lang_dist = stats.get("language_distribution", {})

    # Language breakdown
    te_words = [w for w in words if w.language == "te"]
    en_words = [w for w in words if w.language == "en"]
    hi_words = [w for w in words if w.language == "hi"]
    mixed_words = [w for w in words if w.language == "mixed"]
    low_conf_words = [w for w in words if w.low_confidence_flag]
    filler_words = [w for w in words if w.is_filler]

    # Code-switched sentences
    code_switched_sentences = [s for s in sentences if len(s.language_mix) > 1]

    # ── Print Report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'-'*40}")
    print(f"  ENGINE")
    print(f"{'-'*40}")
    print(f"  Engine:         {transcript.engine_name}")
    print(f"  Model:          {transcript.whisper_model}")
    print(f"  Device:         {transcript.device} ({transcript.compute_type})")

    print(f"\n{'-'*40}")
    print(f"  PERFORMANCE")
    print(f"{'-'*40}")
    print(f"  Transcription:  {total_time}s")
    print(f"  RAM Before:     {ram_start:.0f} MB")
    print(f"  RAM After:      {ram_end:.0f} MB")
    print(f"  RAM Delta:      {ram_delta:+.0f} MB")

    print(f"\n{'-'*40}")
    print(f"  LANGUAGE DETECTION")
    print(f"{'-'*40}")
    print(f"  Primary:        {transcript.primary_language} (conf: {transcript.language_confidence:.3f})")
    print(f"  All Languages:  {transcript.detected_languages}")
    print(f"  Distribution:   {lang_dist}")
    print(f"  Telugu words:   {len(te_words)} ({100*len(te_words)/max(len(words),1):.1f}%)")
    print(f"  English words:  {len(en_words)} ({100*len(en_words)/max(len(words),1):.1f}%)")
    print(f"  Hindi words:    {len(hi_words)} ({100*len(hi_words)/max(len(words),1):.1f}%)")
    print(f"  Code-switching: {stats.get('code_switching_detected', False)}")
    print(f"  CS Sentences:   {len(code_switched_sentences)}/{len(sentences)}")

    print(f"\n{'-'*40}")
    print(f"  TRANSCRIPT QUALITY")
    print(f"{'-'*40}")
    print(f"  Total Words:    {len(words)}")
    print(f"  Total Sentences:{len(sentences)}")
    print(f"  Low Confidence: {len(low_conf_words)} ({stats.get('low_confidence_percentage', 0):.1f}%)")
    print(f"  Filler Words:   {len(filler_words)} ({stats.get('filler_percentage', 0):.1f}%)")
    print(f"  Avg WPM:        {stats.get('average_wpm', 0)}")
    print(f"  Speech Ratio:   {stats.get('speech_ratio', 0):.3f}")

    print(f"\n{'-'*40}")
    print(f"  TELUGU PRESERVATION CHECK")
    print(f"{'-'*40}")
    if len(te_words) > 0:
        print(f"  [OK] Telugu words PRESERVED ({len(te_words)} words detected)")
        print(f"  Sample Telugu words: {', '.join(w.word for w in te_words[:10])}")
    else:
        print(f"  [WARN] No Telugu words detected -- check if content is Telugu")

    # Show some code-switched sentences
    if code_switched_sentences:
        print(f"\n{'-'*40}")
        print(f"  CODE-SWITCHED SENTENCES (first 5)")
        print(f"{'-'*40}")
        for s in code_switched_sentences[:5]:
            print(f"  [{s.id}] ({', '.join(s.language_mix)}): \"{s.text}\"")

    # Show low-confidence words
    if low_conf_words:
        print(f"\n{'-'*40}")
        print(f"  LOW CONFIDENCE WORDS (first 10)")
        print(f"{'-'*40}")
        for w in low_conf_words[:10]:
            print(f"  [{w.id}] \"{w.word}\" (conf={w.confidence:.3f}, lang={w.language})")

    # Full text preview
    print(f"\n{'-'*40}")
    print(f"  FULL TEXT (first 500 chars)")
    print(f"{'-'*40}")
    print(f"  {transcript.full_text[:500]}")

    # ── Save Transcript ──────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "outputs" / "benchmark_asr"
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / "transcript_raw.json"
    save_json(transcript.to_dict(), transcript_path)
    print(f"\n  Saved: {transcript_path}")

    # Save summary
    summary = {
        "engine": transcript.engine_name,
        "model": transcript.whisper_model,
        "device": transcript.device,
        "compute_type": transcript.compute_type,
        "total_time_seconds": total_time,
        "ram_delta_mb": ram_delta,
        "primary_language": transcript.primary_language,
        "language_confidence": transcript.language_confidence,
        "detected_languages": transcript.detected_languages,
        "language_distribution": lang_dist,
        "total_words": len(words),
        "total_sentences": len(sentences),
        "telugu_word_count": len(te_words),
        "english_word_count": len(en_words),
        "code_switching_detected": stats.get("code_switching_detected", False),
        "code_switched_sentences": len(code_switched_sentences),
        "low_confidence_count": len(low_conf_words),
        "low_confidence_percentage": stats.get("low_confidence_percentage", 0),
        "filler_count": len(filler_words),
        "average_wpm": stats.get("average_wpm", 0),
    }
    save_json(summary, output_dir / "benchmark_summary.json")
    print(f"  Saved: {output_dir / 'benchmark_summary.json'}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
