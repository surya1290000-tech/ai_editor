"""
core/asr/fusion.py

Evidence-Based Multi-ASR Hypothesis Fusion Engine
─────────────────────────────────────────────────
Aligns, evaluates, and fuses competing hypotheses from Whisper and
AI4Bharat IndicConformer for multilingual & code-switched Telugu/English speech.

Key Features:
  - Temporal alignment across competing engine time boundaries
  - Multi-factor evidence-based selection (confidence, phonetic consistency,
    English loanword detection, context continuity)
  - Zero simplistic "IndicConformer always wins" rules
  - Retains full candidate audit trail for every word
  - Preserves distinct spoken text, Romanized text, and native script
  - Prevents hallucinated English translations of Telugu words
  - Preserves genuine English technical and conversational terms
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, Tuple

from core.asr.base import (
    ASRWord, ASRSentence, ASRTranscript, ASRCandidate, ASRAlternative,
    detect_script, is_filler_word,
)
from core.asr.transliterator import telugu_word_to_roman
from core.logger import get_logger

logger = get_logger(__name__)

# Common English words that occur in Indian conversational speech
_COMMON_ENGLISH_LOANWORDS = {
    "everyday", "life", "observe", "professional", "world", "imagine",
    "value", "perfection", "beginner", "emotion", "empathy", "replies",
    "care", "doubt", "judge", "reaction", "feeling", "call", "message",
    "video", "reel", "post", "friend", "time", "day", "worst", "best",
    "because", "somebody", "think", "start", "stop", "change", "feel",
    "having", "don't", "dont", "late", "action", "behavior", "attitude",
    "student", "students", "compare", "dance", "western", "class",
}

# Common conversational Telugu words
_CONVERSATIONAL_TELUGU_WORDS = {
    "kani", "kaani", "mana", "okkasari", "chesthe", "ante", "adi", "ala",
    "inka", "cheppanu", "telusa", "untadi", "chestham", "cheppandi", "manaku",
    "nenu", "meeru", "vaadu", "aayana", "aame", "evaru", "enti", "enduku",
    "eppudu", "akkada", "ikkada", "ayindi", "ledu", "unde", "kadhu", "avunu",
    "bagundi", "baguntundi", "choosthe", "chusthe", "raadhu", "vastundi",
    "vellu", "padutundi", "istam", "kashtam", "dhairyam", "badhulu", "prathi",
    "vishayam", "samasya", "janalaki", "lopala", "baita", "mari", "annadu",
    "annadi", "chesaru", "chesindi", "vachindi", "vellindi", "poyindi",
    "teesukunna", "cheppu", "kuda", "endukante", "cheppesi", "arthamavuthundi",
    "untando", "chala", "vatini", "apasanu", "kosam", "wait", "manam",
}


def phonetic_similarity(w1: str, w2: str) -> float:
    """Computes basic phonetic string similarity between two Romanized words."""
    s1 = re.sub(r'[^a-z]', '', w1.lower())
    s2 = re.sub(r'[^a-z]', '', w2.lower())
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if s1 in s2 or s2 in s1:
        return 0.85

    # Levenshtein distance on characters
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    dist = dp[m][n]
    max_len = max(m, n)
    return round(1.0 - (dist / max_len), 3)


class ASRFusionEngine:
    """
    Evidence-based fusion engine aligning and selecting competing hypotheses
    from Whisper and IndicConformer.
    """

    def __init__(
        self,
        time_tolerance_sec: float = 0.40,
        whisper_bias_for_english: float = 0.15,
        indic_bias_for_telugu: float = 0.15,
        time_tolerance_s: Optional[float] = None,
    ):
        if time_tolerance_s is not None:
            time_tolerance_sec = time_tolerance_s
        self.time_tolerance_sec = time_tolerance_sec
        self.whisper_bias_for_english = whisper_bias_for_english
        self.indic_bias_for_telugu = indic_bias_for_telugu

    def fuse_words(
        self,
        whisper_words: List[ASRWord],
        indic_words: List[ASRWord],
    ) -> List[ASRWord]:
        """Aligns and selects competing word hypotheses directly."""
        aligned_pairs = self._temporal_align(whisper_words, indic_words)
        fused = []
        for i, pair in enumerate(aligned_pairs):
            fused.append(self._select_hypothesis(pair.get("whisper"), pair.get("indic"), i))
        return fused

    def fuse(
        self,
        whisper_transcript: ASRTranscript,
        indic_transcript: ASRTranscript,
    ) -> ASRTranscript:
        """
        Fuses Whisper and IndicConformer transcripts into a single high-accuracy transcript.
        """
        w_words = whisper_transcript.words
        i_words = indic_transcript.words

        if not i_words:
            # Fallback to whisper if IndicConformer produced nothing
            for w in w_words:
                w.candidates = [ASRCandidate(engine="whisper", text=w.word, confidence=w.confidence, language=w.language)]
                w.selection_reason = "Whisper-only execution (IndicConformer produced zero words)"
            return whisper_transcript

        if not w_words:
            # Fallback to IndicConformer if Whisper produced nothing
            for w in i_words:
                w.candidates = [ASRCandidate(engine="indic_conformer", text=w.word, confidence=w.confidence, language=w.language)]
                w.selection_reason = "IndicConformer-only execution (Whisper produced zero words)"
            return indic_transcript

        logger.info(
            f"ASRFusion | Aligning {len(w_words)} Whisper words with {len(i_words)} IndicConformer words..."
        )

        # 1. Temporal Cluster Alignment
        aligned_pairs = self._temporal_align(w_words, i_words)

        # 2. Evidence-Based Selection
        fused_words: List[ASRWord] = []
        word_id = 0

        for pair in aligned_pairs:
            w_candidate = pair.get("whisper")
            i_candidate = pair.get("indic")

            selected_word = self._select_hypothesis(w_candidate, i_candidate, word_id)
            fused_words.append(selected_word)
            word_id += 1

        # 3. Construct Sentences from Fused Words
        sentences = self._build_fused_sentences(fused_words)

        # 4. Aggregate Language and Metrics
        detected_languages = sorted(list(set(w.language for w in fused_words)))
        full_text = " ".join(w.word for w in fused_words).strip()
        avg_agreement = round(
            sum(w.agreement_score for w in fused_words) / max(1, len(fused_words)), 3
        )

        fused_transcript = ASRTranscript(
            full_text=full_text,
            detected_languages=detected_languages,
            primary_language="te" if "te" in detected_languages else "en",
            language_confidence=round(sum(w.confidence for w in fused_words) / max(1, len(fused_words)), 3),
            whisper_model=f"Fused({whisper_transcript.whisper_model} + {indic_transcript.whisper_model})",
            words=fused_words,
            sentences=sentences,
            silence_regions=whisper_transcript.silence_regions,
            speech_stats={
                "total_words": len(fused_words),
                "agreement_rate": avg_agreement,
                "telugu_words": sum(1 for w in fused_words if w.language == "te"),
                "english_words": sum(1 for w in fused_words if w.language == "en"),
                "whisper_selected_count": sum(1 for w in fused_words if w.source_engine == "whisper"),
                "indic_selected_count": sum(1 for w in fused_words if w.source_engine == "indic_conformer"),
            },
            engine_name="asr_fusion_engine",
            compute_type="hybrid_fusion",
            device=whisper_transcript.device,
            transcription_time_seconds=round(
                whisper_transcript.transcription_time_seconds + indic_transcript.transcription_time_seconds, 2
            ),
            peak_ram_mb=max(whisper_transcript.peak_ram_mb, indic_transcript.peak_ram_mb),
        )

        logger.info(
            f"ASRFusion | Fusion complete: {len(fused_words)} words, "
            f"Agreement: {avg_agreement:.2f}, Engines: {fused_transcript.speech_stats['whisper_selected_count']} Whisper / "
            f"{fused_transcript.speech_stats['indic_selected_count']} IndicConformer"
        )
        return fused_transcript

    def _temporal_align(
        self,
        whisper_words: List[ASRWord],
        indic_words: List[ASRWord],
    ) -> List[Dict[str, Optional[ASRWord]]]:
        """
        Aligns words from both engines based on time intervals.
        """
        pairs: List[Dict[str, Optional[ASRWord]]] = []
        used_indic_indices = set()

        for w in whisper_words:
            # Find closest overlapping Indic word
            best_match_idx = None
            best_overlap = 0.0

            for idx, i_word in enumerate(indic_words):
                if idx in used_indic_indices:
                    continue

                # Calculate temporal overlap
                overlap_start = max(w.start, i_word.start)
                overlap_end = min(w.end, i_word.end)
                overlap = max(0.0, overlap_end - overlap_start)

                time_dist = abs(w.start - i_word.start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_idx = idx
                elif best_overlap == 0.0 and time_dist <= self.time_tolerance_sec:
                    best_match_idx = idx

            if best_match_idx is not None:
                used_indic_indices.add(best_match_idx)
                pairs.append({
                    "whisper": w,
                    "indic": indic_words[best_match_idx],
                })
            else:
                pairs.append({
                    "whisper": w,
                    "indic": None,
                })

        # Add remaining unmatched Indic words
        for idx, i_word in enumerate(indic_words):
            if idx not in used_indic_indices:
                pairs.append({
                    "whisper": None,
                    "indic": i_word,
                })

        # Sort chronologically by start time
        pairs.sort(key=lambda p: (
            p["whisper"].start if p["whisper"] else p["indic"].start,
            p["whisper"].end if p["whisper"] else p["indic"].end
        ))
        return pairs

    def _select_hypothesis(
        self,
        w: Optional[ASRWord],
        i: Optional[ASRWord],
        word_id: int,
    ) -> ASRWord:
        """
        Evidence-based decision maker choosing between competing hypotheses.
        """
        # Case 1: Only Whisper produced a candidate
        if i is None and w is not None:
            w_clean = w.word.strip()
            rom = telugu_word_to_roman(w_clean) if any('\u0C00' <= c <= '\u0C7F' for c in w_clean) else None
            is_telugu = (w.language == "te") or (rom is not None) or (w_clean.lower() in _CONVERSATIONAL_TELUGU_WORDS)
            return ASRWord(
                id=word_id,
                word=w_clean,
                start=w.start,
                end=w.end,
                duration=w.duration,
                confidence=round(w.confidence, 3),
                language="te" if is_telugu else "en",
                script="latin",
                spoken_text=w_clean,
                romanized_text=rom or w_clean,
                native_script_text=w.native_script_text,
                source_engine="whisper",
                candidates=[ASRCandidate(engine="whisper", text=w_clean, confidence=w.confidence, language=w.language)],
                agreement_score=1.0,
                selection_reason="Single engine coverage: Whisper hypothesis accepted without competition",
            )

        # Case 2: Only IndicConformer produced a candidate
        if w is None and i is not None:
            rom = i.romanized_text or telugu_word_to_roman(i.word)
            is_loanword = rom.lower() in _COMMON_ENGLISH_LOANWORDS
            return ASRWord(
                id=word_id,
                word=rom,
                start=i.start,
                end=i.end,
                duration=i.duration,
                confidence=round(i.confidence, 3),
                language="en" if is_loanword else "te",
                script="latin",
                spoken_text=rom,
                romanized_text=rom,
                native_script_text=i.native_script_text or i.word,
                source_engine="indic_conformer",
                candidates=[ASRCandidate(engine="indic_conformer", text=rom, confidence=i.confidence, language="te")],
                agreement_score=1.0,
                selection_reason="Single engine coverage: IndicConformer hypothesis accepted without competition",
            )

        # Case 3: Both engines produced competing candidates
        w_text = w.word.strip()
        i_native = i.native_script_text or i.word.strip()
        i_rom = i.romanized_text or telugu_word_to_roman(i_native)

        # Record candidates for auditability
        candidates = [
            ASRCandidate(engine="whisper", text=w_text, confidence=round(w.confidence, 3), language=w.language),
            ASRCandidate(engine="indic_conformer", text=i_rom, confidence=round(i.confidence, 3), language="te"),
        ]

        # Calculate phonetic / text similarity
        sim = phonetic_similarity(w_text, i_rom)

        # Evidence checks:
        is_whisper_loanword = w_text.lower() in _COMMON_ENGLISH_LOANWORDS
        is_indic_telugu = i_rom.lower() in _CONVERSATIONAL_TELUGU_WORDS
        whisper_conf = w.confidence
        indic_conf = i.confidence

        # ── Test 1: Consensus (both engines agree phonetically) ──
        if sim >= 0.75:
            # If it's an English loanword, output English spelling
            if is_whisper_loanword:
                chosen_text = w_text.lower()
                chosen_lang = "en"
            else:
                chosen_text = i_rom
                chosen_lang = "te"

            return ASRWord(
                id=word_id,
                word=chosen_text,
                start=w.start,
                end=w.end,
                duration=w.duration,
                confidence=min(0.99, round(max(whisper_conf, indic_conf) + 0.05, 3)),
                language=chosen_lang,
                script="latin",
                spoken_text=chosen_text,
                romanized_text=i_rom,
                native_script_text=i_native,
                source_engine="fused",
                candidates=candidates,
                agreement_score=sim,
                selection_reason=f"Consensus: phonetic similarity ({sim:.2f}) validates token",
            )

        # ── Test 2: English Loanword inside Telugu Speech ─────────
        # e.g., Whisper outputs "observe" (0.90) and IndicConformer outputs "అబ్జర్వ్" (observe)
        if is_whisper_loanword and whisper_conf >= 0.65:
            return ASRWord(
                id=word_id,
                word=w_text,
                start=w.start,
                end=w.end,
                duration=w.duration,
                confidence=round(whisper_conf, 3),
                language="en",
                script="latin",
                spoken_text=w_text,
                romanized_text=w_text,
                native_script_text=i_native,
                source_engine="whisper",
                candidates=candidates,
                agreement_score=sim,
                selection_reason="English loanword confirmed by lexicon and acoustic alignment",
            )

        # ── Test 3: Telugu Phonetic Consistency ──────────────────
        # e.g., IndicConformer outputs "kani" or "mana" or "chesthe"
        if is_indic_telugu and indic_conf >= 0.70:
            return ASRWord(
                id=word_id,
                word=i_rom,
                start=w.start,
                end=w.end,
                duration=w.duration,
                confidence=round(indic_conf, 3),
                language="te",
                script="latin",
                spoken_text=i_rom,
                romanized_text=i_rom,
                native_script_text=i_native,
                source_engine="indic_conformer",
                candidates=candidates,
                agreement_score=sim,
                selection_reason="Conversational Telugu keyword confirmed by IndicConformer phonetics",
            )

        # ── Test 4: Weighted Evidence Decision ────────────────────
        # Compute calibrated score:
        # Score_w = conf + (loanword_prior)
        # Score_i = conf + (telugu_prior)
        score_w = whisper_conf + (self.whisper_bias_for_english if is_whisper_loanword else 0.0)
        score_i = indic_conf + (self.indic_bias_for_telugu if is_indic_telugu else 0.0)

        if score_w >= score_i:
            return ASRWord(
                id=word_id,
                word=w_text,
                start=w.start,
                end=w.end,
                duration=w.duration,
                confidence=round(whisper_conf, 3),
                language=w.language,
                script="latin",
                spoken_text=w_text,
                romanized_text=i_rom if w.language == "te" else w_text,
                native_script_text=i_native,
                source_engine="whisper",
                candidates=candidates,
                agreement_score=sim,
                selection_reason=f"Higher calibrated evidence score (Whisper: {score_w:.2f} vs Indic: {score_i:.2f})",
            )
        else:
            return ASRWord(
                id=word_id,
                word=i_rom,
                start=i.start,
                end=i.end,
                duration=i.duration,
                confidence=round(indic_conf, 3),
                language="te",
                script="latin",
                spoken_text=i_rom,
                romanized_text=i_rom,
                native_script_text=i_native,
                source_engine="indic_conformer",
                candidates=candidates,
                agreement_score=sim,
                selection_reason=f"Higher calibrated evidence score (Indic: {score_i:.2f} vs Whisper: {score_w:.2f})",
            )

    def _build_fused_sentences(self, words: List[ASRWord]) -> List[ASRSentence]:
        """Groups fused words into sentences with language mix metadata."""
        sentences: List[ASRSentence] = []
        if not words:
            return sentences

        cur_word_ids: List[int] = []
        cur_start = words[0].start
        sent_id = 0

        for i, w in enumerate(words):
            cur_word_ids.append(w.id)
            pause_after = (words[i + 1].start - w.end) if i + 1 < len(words) else 1.0

            if pause_after >= 0.40 or len(cur_word_ids) >= 12 or i == len(words) - 1:
                sent_words = [words[wid] for wid in cur_word_ids]
                sent_text = " ".join(sw.word for sw in sent_words)
                sent_dur = round(sent_words[-1].end - cur_start, 3)
                langs = sorted(list(set(sw.language for sw in sent_words)))

                sentences.append(
                    ASRSentence(
                        id=sent_id,
                        text=sent_text,
                        start=round(cur_start, 3),
                        end=round(sent_words[-1].end, 3),
                        duration=sent_dur,
                        word_count=len(sent_words),
                        word_ids=cur_word_ids,
                        language_mix=langs,
                    )
                )
                sent_id += 1
                cur_word_ids = []
                if i + 1 < len(words):
                    cur_start = words[i + 1].start

        return sentences
