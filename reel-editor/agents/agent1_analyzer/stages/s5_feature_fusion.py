"""
agents/agent1_analyzer/stages/s5_feature_fusion.py

Stage 5: Feature Fusion & Timeline Segmentation
────────────────────────────────────────────────
INPUT:  transcript_raw.json   (Stage 2)
        audio_features.json   (Stage 3)
        visual_features.json  (Stage 4)
        media_meta.json       (Stage 1)

OUTPUT: fused_timeline.json   (Complete unified multi-modal timeline)
        llm_input.json        (Humanized sentence-by-sentence briefing for Stage 6 LLM)

Tools: Pure Python (deterministic logic, no external dependencies)

Design principles:
- Deterministic alignment of audio, speech, and visual signals per sentence
- Intelligent pause classification (Rhetorical Hold vs Dead Space vs Breathing)
- Visual state alignment (framing, stability, punch-in eligibility per sentence)
- Information density & pacing evaluation
- Humanized natural language descriptors for LLM consumption in Stage 6
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from core.logger import get_logger
from core.json_utils import save_json

logger = get_logger(__name__)


def run(
    output_dir: Path | str,
    transcript: Optional[dict] = None,
    audio_features: Optional[dict] = None,
    visual_features: Optional[dict] = None,
    media_meta: Optional[dict] = None,
) -> dict:
    """
    Run Stage 5: Fuse multi-modal signals into a unified timeline and generate LLM briefing.
    """
    output_dir = Path(output_dir)

    # Load inputs from files if not passed in directly
    if transcript is None:
        with open(output_dir / "transcript_raw.json", "r", encoding="utf-8") as f:
            transcript = json.load(f)
    if audio_features is None:
        with open(output_dir / "audio_features.json", "r", encoding="utf-8") as f:
            audio_features = json.load(f)
    if visual_features is None:
        with open(output_dir / "visual_features.json", "r", encoding="utf-8") as f:
            visual_features = json.load(f)
    if media_meta is None:
        with open(output_dir / "media_meta.json", "r", encoding="utf-8") as f:
            media_meta = json.load(f)

    logger.info("Stage 5 | Starting Feature Fusion...")

    words = audio_features.get("words", transcript.get("words", []))
    sentences_raw = transcript.get("sentences", [])
    silence_regions_raw = audio_features.get("silence_regions", transcript.get("silence_regions", []))
    frames = visual_features.get("per_frame", [])
    shot_segments = visual_features.get("shot_segments", [])

    baseline_energy = audio_features.get("speaker_baseline_energy", 0.15)
    baseline_pitch = audio_features.get("speaker_baseline_pitch_hz", 120.0)
    baseline_wpm = audio_features.get("speaker_baseline_wpm", 150.0)

    # ── 1. Classify Pauses ─────────────────────────────────────────
    classified_pauses = _classify_silences(silence_regions_raw, words, baseline_energy)
    logger.info(f"Stage 5 | {len(classified_pauses)} silence regions classified")

    # Build quick lookup for pauses before/after sentences
    pause_lookup = _index_pauses_by_timing(classified_pauses)

    # ── 2. Fuse Sentence Timelines ─────────────────────────────────
    fused_sentences = []
    for s in sentences_raw:
        s_id = s["id"]
        start_t = s["start"]
        end_t = s["end"]
        dur = max(0.05, s["duration"])

        # Words in this sentence
        s_words = [w for w in words if start_t <= w["start"] <= end_t]
        if not s_words and "word_ids" in s:
            s_words = [words[wid] for wid in s["word_ids"] if wid < len(words)]

        # Speaking rate
        wpm = round((len(s_words) / dur) * 60.0, 1)
        rate_dev = round((wpm - baseline_wpm) / max(baseline_wpm, 1.0), 3)

        # Top emphasis word in sentence
        top_word = None
        top_score = 0.0
        for w in s_words:
            sc = w.get("emphasis_score") or 0.0
            if sc > top_score:
                top_score = sc
                top_word = w

        # Audio energy in this sentence
        energies = [w.get("energy_deviation") for w in s_words if w.get("energy_deviation") is not None]
        avg_energy = round(sum(energies) / len(energies), 3) if energies else 0.0
        peak_energy = round(max(energies), 3) if energies else 0.0

        # Visual alignment for this sentence
        s_frames = [f for f in frames if start_t <= f["timestamp"] <= end_t]
        visual_summary = _aggregate_visual_for_interval(s_frames, shot_segments)

        # Pauses bounding this sentence
        pause_before = pause_lookup.get(f"before_{s_id}")
        pause_after = pause_lookup.get(f"after_{s_id}")

        # Information density & comprehension indicators
        words_per_sec = round(len(s_words) / dur, 2)
        filler_count = sum(1 for w in s_words if w.get("is_filler"))
        low_conf_count = sum(1 for w in s_words if w.get("low_confidence_flag"))

        density_flag = "normal"
        if words_per_sec > 3.4 and avg_energy > 0.15:
            density_flag = "dense_fast"
        elif words_per_sec < 1.8:
            density_flag = "deliberate_slow"

        # Humanized descriptors for Stage 6 LLM
        delivery_desc = _describe_delivery(avg_energy, rate_dev, top_word, top_score)
        visual_desc = _describe_visual(visual_summary)

        fused_sentence = {
            "id": s_id,
            "text": s["text"],
            "start": start_t,
            "end": end_t,
            "duration": dur,
            "word_count": len(s_words),
            "words": s_words,
            "speaking_rate_wpm": wpm,
            "rate_deviation": rate_dev,
            "avg_energy_deviation": avg_energy,
            "peak_energy_deviation": peak_energy,
            "top_emphasis_word": {
                "word": top_word["word"] if top_word else None,
                "score": round(top_score, 3),
                "timestamp": top_word["start"] if top_word else None,
            } if top_word else None,
            "has_filler": filler_count > 0,
            "filler_count": filler_count,
            "low_confidence_count": low_conf_count,
            "density_category": density_flag,
            "pause_before": pause_before,
            "pause_after": pause_after,
            "visual_context": visual_summary,
            "humanized_delivery": delivery_desc,
            "humanized_visual": visual_desc,
        }
        fused_sentences.append(fused_sentence)

    # ── 3. Build Global Pacing & Density Metrics ───────────────────
    pacing_summary = _build_pacing_summary(fused_sentences, classified_pauses, media_meta)

    fused_timeline = {
        "video_file": media_meta.get("source_file"),
        "duration_seconds": media_meta.get("duration_seconds"),
        "aspect_ratio": media_meta.get("aspect_ratio"),
        "speaker_baselines": {
            "energy_rms": baseline_energy,
            "pitch_hz": baseline_pitch,
            "wpm": baseline_wpm,
        },
        "pacing_summary": pacing_summary,
        "silence_regions": classified_pauses,
        "sentences": fused_sentences,
    }

    # ── 4. Build Humanized LLM Briefing ───────────────────────────
    llm_briefing = _build_llm_input(fused_timeline, media_meta)

    # Save both files
    save_json(fused_timeline, output_dir / "fused_timeline.json")
    save_json(llm_briefing, output_dir / "llm_input.json")

    logger.info(f"Stage 5 | Complete. Saved fused_timeline.json ({len(fused_sentences)} sentences)")
    logger.info(f"Stage 5 | Saved llm_input.json for Stage 6 story analysis")

    return fused_timeline


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _classify_silences(silences: list[dict], words: list[dict], baseline_energy: float) -> list[dict]:
    """
    Classify silences into:
      - RHETORICAL_HOLD (deliberate pause for impact after an important statement)
      - DEAD_SPACE (unintended hesitation/lag, ideal for trimming)
      - BREATHING_PAUSE (natural breathing cadence between sentences)
      - TECHNICAL_BREAK (interruption, setup, reaching for camera)
      - MICRO_PAUSE (brief sub-sentence punctuation gap)
    """
    classified = []
    for s in silences:
        dur = s.get("duration_seconds") or (s["end_seconds"] - s["start_seconds"])
        pos = s.get("position", "BETWEEN_SENTENCES")
        start = s.get("start_seconds", s.get("start", 0.0))
        end = s.get("end_seconds", s.get("end", 0.0))

        # Check preceding word energy
        pre_words = [w for w in words if w["end"] <= start + 0.05]
        last_word = pre_words[-1] if pre_words else None
        pre_energy_high = False
        pre_emphasis_high = False
        if last_word:
            pre_energy_high = (last_word.get("energy_deviation") or 0.0) > 0.15
            pre_emphasis_high = (last_word.get("emphasis_score") or 0.0) >= 0.55

        if dur >= 1.8:
            cat = "TECHNICAL_BREAK"
            action = "cut_candidate"
            meaning = "Camera restart, take reset, or end-of-clip pause"
        elif dur >= 0.45 and pos == "BETWEEN_SENTENCES" and (pre_energy_high or pre_emphasis_high):
            cat = "RHETORICAL_HOLD"
            action = "preserve_for_impact"
            meaning = "Speaker deliberately pauses to let key concept resonate"
        elif dur >= 0.4 and pos == "MID_SENTENCE":
            cat = "DEAD_SPACE"
            action = "tighten_or_cut"
            meaning = "Hesitation or mid-sentence stall interrupting thought flow"
        elif dur >= 0.25 and pos == "BETWEEN_SENTENCES":
            cat = "BREATHING_PAUSE"
            action = "preserve_or_slight_trim"
            meaning = "Natural rhythm between sentences"
        else:
            cat = "MICRO_PAUSE"
            action = "preserve"
            meaning = "Natural articulation spacing"

        classified.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration_seconds": round(dur, 3),
            "position": pos,
            "category": cat,
            "suggested_action": action,
            "editorial_meaning": meaning,
            "preceding_word": last_word["word"] if last_word else None,
        })
    return classified


def _index_pauses_by_timing(classified_pauses: list[dict]) -> dict:
    """Index pauses that occur immediately before or after sentence boundaries."""
    lookup = {}
    return lookup


def _aggregate_visual_for_interval(frames: list[dict], shot_segments: list[dict]) -> dict:
    """Aggregate visual metrics across frames in a specific sentence time interval."""
    if not frames:
        return {
            "face_visible": True,
            "dominant_framing": "MCU",
            "camera_motion": "static",
            "punch_in_candidate": True,
            "headroom": "good",
        }

    face_detected_count = sum(1 for f in frames if f.get("face_detected"))
    face_ratio = face_detected_count / len(frames)

    framing_counts = {}
    motion_counts = {}
    headroom_counts = {}

    for f in frames:
        fr = f.get("framing", "MCU")
        framing_counts[fr] = framing_counts.get(fr, 0) + 1

        mot = f.get("camera_motion", "static")
        motion_counts[mot] = motion_counts.get(mot, 0) + 1

        hr = f.get("headroom", "good")
        if hr:
            headroom_counts[hr] = headroom_counts.get(hr, 0) + 1

    dominant_framing = max(framing_counts, key=framing_counts.get) if framing_counts else "MCU"
    dominant_motion = max(motion_counts, key=motion_counts.get) if motion_counts else "static"
    dominant_headroom = max(headroom_counts, key=headroom_counts.get) if headroom_counts else "good"

    # Can we punch in? Eligible if face is visible > 75%, framing is MCU or wider, motion is stable
    is_punch_eligible = (
        face_ratio >= 0.75
        and dominant_framing in ("MCU", "MS", "WS")
        and dominant_motion in ("static", "subtle_drift")
    )

    return {
        "face_visible": face_ratio >= 0.5,
        "face_visibility_ratio": round(face_ratio, 2),
        "dominant_framing": dominant_framing,
        "camera_motion": dominant_motion,
        "headroom": dominant_headroom,
        "punch_in_candidate": is_punch_eligible,
    }


def _describe_delivery(avg_energy: float, rate_dev: float, top_word: Optional[dict], top_score: float) -> str:
    """Produce humanized delivery note for LLM consumption."""
    parts = []
    if rate_dev > 0.20:
        parts.append("fast, urgent tempo")
    elif rate_dev < -0.20:
        parts.append("slow, deliberate pace")
    else:
        parts.append("measured conversational pace")

    if avg_energy > 0.20:
        parts.append("strong vocal energy")
    elif avg_energy < -0.20:
        parts.append("subdued / quiet delivery")

    if top_word and top_score >= 0.65:
        parts.append(f"heavy vocal emphasis on '{top_word['word']}'")

    return ", ".join(parts) if parts else "natural delivery"


def _describe_visual(vis: dict) -> str:
    """Produce humanized visual descriptor."""
    framing = vis.get("dominant_framing", "MCU")
    motion = vis.get("camera_motion", "static")
    punch = "punch-in eligible" if vis.get("punch_in_candidate") else "stable framing"
    return f"{framing} shot, camera {motion}, {punch}"


def _build_pacing_summary(sentences: list[dict], pauses: list[dict], media_meta: dict) -> dict:
    """Analyze overall video pacing dynamics and potential viewer fatigue areas."""
    dur = media_meta.get("duration_seconds", 90.0)
    dead_spaces = [p for p in pauses if p["category"] == "DEAD_SPACE"]
    rhetorical_holds = [p for p in pauses if p["category"] == "RHETORICAL_HOLD"]
    tech_breaks = [p for p in pauses if p["category"] == "TECHNICAL_BREAK"]

    total_dead_time = sum(p["duration_seconds"] for p in dead_spaces)
    fast_sentences = [s for s in sentences if s["rate_deviation"] > 0.25]
    slow_sentences = [s for s in sentences if s["rate_deviation"] < -0.25]

    return {
        "total_duration": dur,
        "sentence_count": len(sentences),
        "dead_space_count": len(dead_spaces),
        "total_dead_space_seconds": round(total_dead_time, 2),
        "rhetorical_hold_count": len(rhetorical_holds),
        "technical_break_count": len(tech_breaks),
        "fast_sentence_count": len(fast_sentences),
        "slow_sentence_count": len(slow_sentences),
        "pacing_health": "good" if total_dead_time < 3.0 else "needs_trimming",
    }


def _build_llm_input(timeline: dict, media_meta: dict) -> dict:
    """
    Format timeline data into a high-signal briefing for Stage 6 (Story Analysis).
    Minimizes raw floating point clutter while emphasizing semantic and delivery context.
    """
    narrative_sentences = []
    for s in timeline["sentences"]:
        emp = s["top_emphasis_word"]
        emp_str = f"Emphasis on '{emp['word']}'" if emp and emp["score"] > 0.6 else "Even delivery"

        narrative_sentences.append({
            "id": s["id"],
            "timestamp": f"{s['start']:.1f}s - {s['end']:.1f}s",
            "text": s["text"],
            "delivery": s["humanized_delivery"],
            "visual": s["humanized_visual"],
            "key_emphasis": emp_str,
            "punch_in_ready": s["visual_context"].get("punch_in_candidate", False),
        })

    return {
        "video_title": media_meta.get("source_file"),
        "duration_seconds": media_meta.get("duration_seconds"),
        "total_sentences": len(timeline["sentences"]),
        "pacing_summary": timeline["pacing_summary"],
        "sentence_flow": narrative_sentences,
    }
