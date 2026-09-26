"""
core/evidence/collector.py

Standardized Multi-Modal Evidence Collector (Phase 2)
────────────────────────────────────────────────────
Transforms Stage 1–5 multi-modal analysis into:
  1. EnrichedTranscript (word-level acoustic context, confidence, uncertainty)
  2. EvidenceInventory (standardized evidence items with strict observation-only rules)

Key Design Rules:
  - Agent 1 is an OBSERVER answering "What is happening in this footage?"
  - NO creative decisions or styling commands.
  - NO_EDIT regions are first-class evidence items.
  - Low-confidence ASR words propagate uncertainty and cap semantic confidence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from core.logger import get_logger
from core.evidence.models import (
    EnrichedWord,
    EnrichedSentence,
    EnrichedTranscript,
    EvidenceSource,
    EvidenceSignals,
    EvidenceItem,
    EvidenceInventory,
)

logger = get_logger(__name__)


class EvidenceCollector:
    """Collects and standardizes multimodal evidence from Stage 1–5 outputs."""

    def __init__(
        self,
        media_meta: dict[str, Any],
        transcript_raw: dict[str, Any],
        audio_features: dict[str, Any],
        visual_features: dict[str, Any],
        fused_timeline: dict[str, Any],
    ):
        self.media_meta = media_meta
        self.transcript_raw = transcript_raw
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.fused_timeline = fused_timeline

        self.duration = float(media_meta.get("duration_seconds", 0.0))
        self.source_video = str(media_meta.get("source_file", "video.mp4"))

        self._evidence_counter = 0

    def _next_id(self) -> str:
        self._evidence_counter += 1
        return f"ev_{self._evidence_counter:03d}"

    def build_enriched_transcript(self) -> EnrichedTranscript:
        """Enriches raw transcript with acoustic features, visual alignment, and uncertainty flags."""
        raw_words = self.transcript_raw.get("words", [])
        af_words = {w["id"]: w for w in self.audio_features.get("words", [])}

        enriched_words: list[EnrichedWord] = []
        for w in raw_words:
            wid = w["id"]
            af_w = af_words.get(wid, {})

            conf = float(w.get("confidence", 0.8))
            lang = w.get("language", "und")
            script = w.get("script", "latin")
            romanized = w.get("romanized", w["word"])

            # Acoustic enhancements from Stage 3
            energy_dev = float(af_w.get("energy_deviation", w.get("energy_deviation", 0.0)))
            pitch_dev = float(af_w.get("pitch_deviation", w.get("pitch_deviation", 0.0)))
            emphasis_score = float(af_w.get("emphasis_score", w.get("emphasis_score", 0.0)))
            is_filler = bool(w.get("is_filler", False))

            # Uncertainty assessment
            uncertainty_flags: list[str] = []
            low_conf_flag = conf < 0.60
            if low_conf_flag:
                uncertainty_flags.append(f"low_asr_confidence:{conf:.2f}")
            if lang == "te" and script == "latin":
                uncertainty_flags.append("romanized_telugu_transcription")
            if conf < 0.40:
                uncertainty_flags.append("potential_phonetic_hallucination")

            ew = EnrichedWord(
                id=wid,
                word=w["word"],
                start=float(w["start"]),
                end=float(w["end"]),
                duration=float(w.get("duration", w["end"] - w["start"])),
                confidence=conf,
                language=lang,
                script=script,
                romanized=romanized,
                is_filler=is_filler,
                emphasis_score=emphasis_score,
                energy_deviation=energy_dev,
                pitch_deviation=pitch_dev,
                low_confidence_flag=low_conf_flag,
                alternate_hypotheses=w.get("alternate_hypotheses", []),
                uncertainty=uncertainty_flags,
            )
            enriched_words.append(ew)

        # Build Enriched Sentences
        fused_sentences = self.fused_timeline.get("sentences", [])
        enriched_sentences: list[EnrichedSentence] = []

        words_by_id = {w.id: w for w in enriched_words}

        for s in fused_sentences:
            sid = s["id"]
            st = float(s["start"])
            en = float(s["end"])
            dur = max(0.01, en - st)

            # Gather sentence words
            s_word_ids = [w.id for w in enriched_words if st <= w.start <= en]
            if not s_word_ids and "words" in s:
                s_word_ids = [w["id"] for w in s["words"] if w["id"] in words_by_id]

            s_words = [words_by_id[wid] for wid in s_word_ids if wid in words_by_id]

            confidences = [w.confidence for w in s_words]
            avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 1.0
            low_conf_c = sum(1 for w in s_words if w.low_confidence_flag)
            filler_c = sum(1 for w in s_words if w.is_filler)

            lang_mix = sorted(list({w.language for w in s_words if w.language != "und"}))

            # Aggregate uncertainty
            sent_uncertainty: list[str] = []
            if low_conf_c > 0:
                sent_uncertainty.append(f"{low_conf_c}_words_with_low_confidence")
            if avg_conf < 0.65:
                sent_uncertainty.append(f"sentence_avg_confidence_low:{avg_conf:.2f}")
            if len(lang_mix) > 1:
                sent_uncertainty.append(f"code_switched_mix:{'+'.join(lang_mix)}")

            visual_ctx = s.get("visual_context", {})

            es = EnrichedSentence(
                id=sid,
                text=s["text"],
                start=st,
                end=en,
                duration=dur,
                language_mix=lang_mix,
                word_ids=s_word_ids,
                speaking_rate_wpm=float(s.get("speaking_rate_wpm", 0.0)),
                avg_energy_deviation=float(s.get("avg_energy_deviation", 0.0)),
                peak_energy_deviation=float(s.get("peak_energy_deviation", 0.0)),
                avg_confidence=avg_conf,
                low_confidence_count=low_conf_c,
                filler_count=filler_c,
                top_emphasis_word=s.get("top_emphasis_word"),
                visual_context=visual_ctx,
                uncertainty=sent_uncertainty,
            )
            enriched_sentences.append(es)

        speech_stats = self.transcript_raw.get("speech_stats", {})
        primary_lang = speech_stats.get("detected_language", "und")
        all_langs = self.transcript_raw.get("detected_languages", [primary_lang])

        return EnrichedTranscript(
            version="3.0.0",
            source_audio=str(self.media_meta.get("audio_path", "")),
            duration_seconds=self.duration,
            primary_language=primary_lang,
            detected_languages=all_langs,
            words=enriched_words,
            sentences=enriched_sentences,
            speech_stats=speech_stats,
        )

    def collect_evidence_inventory(
        self,
        enriched_transcript: EnrichedTranscript,
    ) -> EvidenceInventory:
        """
        Extracts all multi-modal evidence items across the video.
        Guarantees strict separation between observation and decision.
        Includes first-class NO_EDIT regions.
        """
        self._evidence_counter = 0
        evidence_list: list[EvidenceItem] = []

        words_by_id = {w.id: w for w in enriched_transcript.words}
        sentences = enriched_transcript.sentences

        # ── 1. Hook Candidate (First 3-8s) ─────────────────────────────
        hook_end = min(self.duration, 8.5)
        hook_words = [w for w in enriched_transcript.words if w.start < hook_end]
        hook_word_ids = [w.id for w in hook_words]
        hook_sents = [s.id for s in sentences if s.start < hook_end]

        hook_conf = round(sum(w.confidence for w in hook_words) / max(len(hook_words), 1), 3)
        hook_unc: list[str] = []
        low_hook_words = [w for w in hook_words if w.confidence < 0.50]
        if low_hook_words:
            hook_unc.append(f"{len(low_hook_words)}_words_with_low_confidence")
            for lw in low_hook_words[:3]:
                hook_unc.append(f"low_asr_confidence:{lw.word}:{lw.confidence:.2f}")
            hook_conf = min(hook_conf, 0.75)
        elif hook_conf < 0.65:
            hook_unc.append(f"low_hook_transcription_confidence:{hook_conf:.2f}")

        evidence_list.append(
            EvidenceItem(
                evidence_id=self._next_id(),
                type="hook_candidate",
                start=0.0,
                end=hook_end,
                source=EvidenceSource(transcript_segment_ids=hook_sents, word_ids=hook_word_ids),
                observation="Opening statement delivery establishing premise and core thesis topic.",
                signals=EvidenceSignals(
                    speech={
                        "text_preview": enriched_transcript.sentences[0].text if enriched_transcript.sentences else "",
                        "language_mix": enriched_transcript.sentences[0].language_mix if enriched_transcript.sentences else [],
                        "wpm": enriched_transcript.sentences[0].speaking_rate_wpm if enriched_transcript.sentences else 0.0,
                    },
                    audio={"baseline_energy": self.audio_features.get("speaker_baseline_energy", 0.0)},
                    visual={"dominant_framing": self.visual_features.get("dominant_framing", "MCU")},
                ),
                confidence=hook_conf,
                reliability="high" if hook_conf > 0.8 and not hook_unc else "medium",
                uncertainty=hook_unc,
                tags=["hook", "opening", "thesis_init"],
            )
        )

        # ── 2. Silence & Pause Regions (Dead Space vs Rhetorical Pause) ──
        silences = self.fused_timeline.get("silence_regions", [])
        for p in silences:
            st = float(p.get("start", 0.0))
            en = float(p.get("end", 0.0))
            dur = float(p.get("duration_seconds", en - st))
            cat = p.get("category", "")
            pre_word = p.get("preceding_word")

            # Check preceding word in enriched transcript
            pre_w_obj = None
            for w in enriched_transcript.words:
                if abs(w.end - st) < 0.15:
                    pre_w_obj = w
                    break

            unc: list[str] = []
            if pre_w_obj and pre_w_obj.low_confidence_flag:
                unc.append(f"preceding_word_low_confidence:{pre_w_obj.confidence:.2f}")

            if cat == "DEAD_SPACE" and dur >= 0.35:
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="dead_space",
                        start=st,
                        end=en,
                        source=EvidenceSource(
                            transcript_segment_ids=[],
                            word_ids=[pre_w_obj.id] if pre_w_obj else [],
                        ),
                        observation=f"Hesitation pause of {dur:.2f}s with no semantic hold or vocal decay.",
                        signals=EvidenceSignals(
                            audio={"pause_duration": dur, "position": p.get("position", "")},
                            speech={"preceding_word": pre_word},
                            visual={"face_tracked": True},
                        ),
                        confidence=0.75 if unc else 0.92,
                        reliability="high" if not unc else "medium",
                        uncertainty=unc,
                        tags=["pause", "dead_space", "pacing_drag"],
                    )
                )
            elif cat == "RHETORICAL_HOLD" and dur >= 0.40:
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="rhetorical_pause",
                        start=st,
                        end=en,
                        source=EvidenceSource(
                            transcript_segment_ids=[],
                            word_ids=[pre_w_obj.id] if pre_w_obj else [],
                        ),
                        observation=f"Deliberate rhetorical pause of {dur:.2f}s allowing preceding statement to resonate.",
                        signals=EvidenceSignals(
                            audio={
                                "pause_duration": dur,
                                "preceding_energy": pre_w_obj.energy_deviation if pre_w_obj else 0.0,
                            },
                            speech={"preceding_word": pre_word},
                            visual={"face_tracked": True},
                        ),
                        confidence=0.75 if unc else 0.88,
                        reliability="high" if not unc else "medium",
                        uncertainty=unc,
                        tags=["pause", "rhetorical_hold", "emphasis"],
                    )
                )

        # ── 3. Filler Words ────────────────────────────────────────────
        for w in enriched_transcript.words:
            if w.is_filler:
                unc = list(w.uncertainty)
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="filler",
                        start=w.start,
                        end=w.end,
                        source=EvidenceSource(transcript_segment_ids=[], word_ids=[w.id]),
                        observation=f"Spoken filler vocalization: '{w.word}'.",
                        signals=EvidenceSignals(
                            speech={"word": w.word, "language": w.language, "confidence": w.confidence},
                            audio={"energy_deviation": w.energy_deviation},
                            visual={},
                        ),
                        confidence=w.confidence,
                        reliability="high" if w.confidence > 0.8 and not unc else "medium",
                        uncertainty=unc,
                        tags=["speech", "filler"],
                    )
                )

        # ── 4. Sentence Boundaries & Delivery Assessment ───────────────
        for s in sentences:
            # Sentence boundary evidence
            s_unc = list(s.uncertainty)
            evidence_list.append(
                EvidenceItem(
                    evidence_id=self._next_id(),
                    type="sentence_boundary",
                    start=s.start,
                    end=s.end,
                    source=EvidenceSource(transcript_segment_ids=[s.id], word_ids=s.word_ids),
                    observation=f"Complete sentence unit (WPM: {s.speaking_rate_wpm:.1f}, mix: {s.language_mix}): \"{s.text}\"",
                    signals=EvidenceSignals(
                        speech={
                            "text": s.text,
                            "wpm": s.speaking_rate_wpm,
                            "word_count": len(s.word_ids),
                            "language_mix": s.language_mix,
                        },
                        audio={
                            "avg_energy_deviation": s.avg_energy_deviation,
                            "peak_energy_deviation": s.peak_energy_deviation,
                        },
                        visual=s.visual_context,
                    ),
                    confidence=s.avg_confidence,
                    reliability="high" if s.avg_confidence > 0.8 and not s_unc else "medium",
                    uncertainty=s_unc,
                    tags=["sentence", "narrative_flow"],
                )
            )

            # Delivery Strength
            if s.avg_energy_deviation > 0.15 and s.speaking_rate_wpm > 130:
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="strong_delivery",
                        start=s.start,
                        end=s.end,
                        source=EvidenceSource(transcript_segment_ids=[s.id], word_ids=s.word_ids),
                        observation=f"High-energy vocal delivery with assertive pacing ({s.speaking_rate_wpm:.0f} WPM).",
                        signals=EvidenceSignals(
                            speech={"wpm": s.speaking_rate_wpm},
                            audio={"avg_energy": s.avg_energy_deviation, "peak_energy": s.peak_energy_deviation},
                            visual=s.visual_context,
                        ),
                        confidence=s.avg_confidence,
                        reliability="high" if s.avg_confidence > 0.8 else "medium",
                        uncertainty=s_unc,
                        tags=["delivery", "vocal_energy", "strong"],
                    )
                )
            elif s.avg_energy_deviation < -0.15 or s.speaking_rate_wpm < 100:
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="weak_delivery",
                        start=s.start,
                        end=s.end,
                        source=EvidenceSource(transcript_segment_ids=[s.id], word_ids=s.word_ids),
                        observation=f"Subdued vocal energy or sluggish delivery pace ({s.speaking_rate_wpm:.0f} WPM).",
                        signals=EvidenceSignals(
                            speech={"wpm": s.speaking_rate_wpm},
                            audio={"avg_energy": s.avg_energy_deviation},
                            visual=s.visual_context,
                        ),
                        confidence=s.avg_confidence,
                        reliability="medium" if s.avg_confidence > 0.6 else "low",
                        uncertainty=s_unc,
                        tags=["delivery", "vocal_energy", "subdued"],
                    )
                )

        # ── 5. Word-level Vocal Emphasis Peaks ─────────────────────────
        for w in enriched_transcript.words:
            if w.emphasis_score >= 0.55 and not w.is_filler:
                unc = list(w.uncertainty)
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="emphasis",
                        start=w.start,
                        end=w.end,
                        source=EvidenceSource(transcript_segment_ids=[], word_ids=[w.id]),
                        observation=f"Acoustic emphasis peak on word '{w.word}' (score={w.emphasis_score:.2f}, energy_dev={w.energy_deviation:+.2f}).",
                        signals=EvidenceSignals(
                            speech={"word": w.word, "language": w.language, "script": w.script},
                            audio={
                                "emphasis_score": w.emphasis_score,
                                "energy_deviation": w.energy_deviation,
                                "pitch_deviation": w.pitch_deviation,
                            },
                            visual={"strong_signal": True},
                        ),
                        confidence=w.confidence,
                        reliability="high" if w.confidence > 0.8 and not unc else "medium",
                        uncertainty=unc,
                        tags=["emphasis", "vocal_peak"],
                    )
                )

        # ── 6. Visual Changes, Framing & Motion ────────────────────────
        shot_segments = self.visual_features.get("shot_segments", [])
        for seg in shot_segments:
            st = float(seg.get("start", 0.0))
            en = float(seg.get("end", 0.0))
            framing = seg.get("framing", "MCU")
            stability = float(seg.get("stability_score", 0.9))

            evidence_list.append(
                EvidenceItem(
                    evidence_id=self._next_id(),
                    type="framing_change" if seg != shot_segments[0] else "visual_change",
                    start=st,
                    end=en,
                    source=EvidenceSource(transcript_segment_ids=[], word_ids=[]),
                    observation=f"Visual framing segment maintaining dominant framing '{framing}' (stability: {stability:.2f}).",
                    signals=EvidenceSignals(
                        speech={},
                        audio={},
                        visual={"framing": framing, "stability_score": stability, "shot_id": seg.get("id")},
                    ),
                    confidence=0.92,
                    reliability="high",
                    uncertainty=[],
                    tags=["visual", "framing", framing.lower()],
                )
            )

        # ── 7. Editing Opportunities (Punch-In, B-roll, SFX) ────────────
        # Punch-In Opportunities: Points of strong vocal emphasis with stable face framing
        emphasis_items = [e for e in evidence_list if e.type == "emphasis"]
        for emp in emphasis_items:
            # Check visual stability around this moment
            emp_w_id = emp.source.word_ids[0] if emp.source.word_ids else None
            emp_word = words_by_id.get(emp_w_id) if emp_w_id is not None else None

            # Look up sentence visual context
            visual_ok = True
            for s in sentences:
                if s.start <= emp.start <= s.end:
                    if s.visual_context.get("punch_in_eligible") is False:
                        visual_ok = False
                    break

            if visual_ok and emp.confidence >= 0.70:
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="punch_in_opportunity",
                        start=emp.start,
                        end=min(self.duration, emp.end + 2.0),
                        source=EvidenceSource(transcript_segment_ids=[], word_ids=emp.source.word_ids),
                        observation=f"Vocal emphasis peak on '{emp_word.word if emp_word else 'keyword'}' with stable head framing; visual emphasis candidate.",
                        signals=EvidenceSignals(
                            speech={"keyword": emp_word.word if emp_word else ""},
                            audio=emp.signals.audio,
                            visual={"punch_in_eligible": True, "framing": "MCU"},
                        ),
                        confidence=emp.confidence,
                        reliability=emp.reliability,
                        uncertainty=list(emp.uncertainty),
                        tags=["opportunity", "visual_emphasis"],
                    )
                )

                # SFX Opportunity at the same vocal punch moment
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="sfx_opportunity",
                        start=emp.start,
                        end=emp.end,
                        source=EvidenceSource(transcript_segment_ids=[], word_ids=emp.source.word_ids),
                        observation=f"Sharply articulated emphasis peak on '{emp_word.word if emp_word else 'keyword'}'; acoustic punctuation candidate.",
                        signals=EvidenceSignals(
                            speech={"keyword": emp_word.word if emp_word else ""},
                            audio=emp.signals.audio,
                            visual={},
                        ),
                        confidence=emp.confidence,
                        reliability=emp.reliability,
                        uncertainty=list(emp.uncertainty),
                        tags=["opportunity", "sound_accent"],
                    )
                )

        # B-Roll Opportunities: Sentences describing vivid, external concepts
        conceptual_keywords = {
            "world", "empathy", "life", "podcast", "book", "day", "people",
            "judge", "asking", "hurt", "stay", "problem"
        }
        for s in sentences:
            s_words_lower = {w.word.lower().strip(".,!?:;\"") for w in enriched_transcript.words if w.id in s.word_ids}
            matched_concepts = s_words_lower.intersection(conceptual_keywords)
            if len(matched_concepts) >= 2 and s.duration >= 2.5:
                s_unc = list(s.uncertainty)
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="broll_opportunity",
                        start=s.start,
                        end=s.end,
                        source=EvidenceSource(transcript_segment_ids=[s.id], word_ids=s.word_ids),
                        observation=f"Abstract conceptual discussion referencing ({', '.join(sorted(matched_concepts))}); candidate for illustrative context overlay.",
                        signals=EvidenceSignals(
                            speech={"concepts": sorted(list(matched_concepts)), "text": s.text},
                            audio={"wpm": s.speaking_rate_wpm},
                            visual={"dominant_framing": "MCU"},
                        ),
                        confidence=min(0.88, s.avg_confidence),
                        reliability="high" if s.avg_confidence > 0.8 and not s_unc else "medium",
                        uncertainty=s_unc,
                        tags=["opportunity", "broll", "illustrative"],
                    )
                )

        # ── 8. First-Class NO_EDIT Regions ─────────────────────────────
        # Identify intervals of fluent, uninterrupted, stable delivery
        # where no dead space, no excessive fillers, and good speech cadence occur.
        dead_space_times = [
            (e.start, e.end) for e in evidence_list if e.type == "dead_space"
        ]

        for s in sentences:
            if s.duration >= 2.5 and s.filler_count == 0 and s.speaking_rate_wpm >= 120 and s.avg_confidence >= 0.75:
                # Ensure no dead space inside this sentence
                has_dead_space = any(
                    not (ds_en <= s.start or ds_st >= s.end)
                    for ds_st, ds_en in dead_space_times
                )
                if not has_dead_space:
                    s_unc = list(s.uncertainty)
                    evidence_list.append(
                        EvidenceItem(
                            evidence_id=self._next_id(),
                            type="no_edit_region",
                            start=s.start,
                            end=s.end,
                            source=EvidenceSource(transcript_segment_ids=[s.id], word_ids=s.word_ids),
                            observation="Natural vocal delivery and steady visual cadence; continuous flow requires no editorial intervention.",
                            signals=EvidenceSignals(
                                speech={"wpm": s.speaking_rate_wpm, "fluency": "high"},
                                audio={"avg_energy": s.avg_energy_deviation},
                                visual={"face_stable": True},
                            ),
                            confidence=s.avg_confidence,
                            reliability="high" if not s_unc else "medium",
                            uncertainty=s_unc,
                            tags=["no_edit", "natural_delivery", "continuity"],
                        )
                    )

        # Ensure no_edit_region items exist even for single continuous monologues
        if not any(e.type == "no_edit_region" for e in evidence_list):
            last_t = 0.0
            sorted_dead_spaces = sorted(dead_space_times, key=lambda x: x[0])
            for ds_st, ds_en in sorted_dead_spaces:
                if ds_st - last_t >= 2.5:
                    evidence_list.append(
                        EvidenceItem(
                            evidence_id=self._next_id(),
                            type="no_edit_region",
                            start=round(last_t, 3),
                            end=round(ds_st, 3),
                            source=EvidenceSource(transcript_segment_ids=[], word_ids=[]),
                            observation="Uninterrupted natural speech span between pauses; delivery continuity requires preservation.",
                            signals=EvidenceSignals(
                                speech={"fluency": "high"},
                                audio={"level": "normal"},
                                visual={"face_stable": True},
                            ),
                            confidence=0.88,
                            reliability="high",
                            uncertainty=[],
                            tags=["no_edit", "natural_delivery", "continuity"],
                        )
                    )
                last_t = max(last_t, ds_en)
            dur_total = self.duration if self.duration > 0 else 60.0
            if dur_total - last_t >= 2.5:
                evidence_list.append(
                    EvidenceItem(
                        evidence_id=self._next_id(),
                        type="no_edit_region",
                        start=round(last_t, 3),
                        end=round(dur_total, 3),
                        source=EvidenceSource(transcript_segment_ids=[], word_ids=[]),
                        observation="Final natural speech span; delivery continuity requires preservation.",
                        signals=EvidenceSignals(
                            speech={"fluency": "high"},
                            audio={"level": "normal"},
                            visual={"face_stable": True},
                        ),
                        confidence=0.88,
                        reliability="high",
                        uncertainty=[],
                        tags=["no_edit", "natural_delivery", "continuity"],
                    )
                )

        # Propagate low ASR confidence to any evidence items referencing those words
        for ev in evidence_list:
            if ev.source.word_ids:
                ref_words = [words_by_id[wid] for wid in ev.source.word_ids if wid in words_by_id]
                low_conf_words = [w for w in ref_words if w.confidence < 0.50]
                if low_conf_words:
                    for lcw in low_conf_words:
                        flag = f"low_asr_confidence:{lcw.word}:{lcw.confidence:.2f}"
                        if flag not in ev.uncertainty:
                            ev.uncertainty.append(flag)
                    has_strong = bool(ev.signals.visual.get("strong_signal")) or bool(ev.signals.audio.get("strong_signal"))
                    if not has_strong:
                        ev.confidence = min(0.75, ev.confidence)

        # Sort evidence items chronologically
        evidence_list.sort(key=lambda x: (x.start, x.end, x.evidence_id))

        return EvidenceInventory(
            version="3.0.0",
            source_video=self.source_video,
            duration_seconds=self.duration,
            evidence=evidence_list,
            metadata={
                "total_items": len(evidence_list),
                "types_count": {
                    t: sum(1 for e in evidence_list if e.type == t)
                    for t in set(e.type for e in evidence_list)
                },
            },
        )
