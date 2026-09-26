"""
core/asr/whisper_engine.py

Multilingual Whisper ASR Engine
───────────────────────────────
Wraps faster-whisper (CTranslate2 backend) with:
  - Configurable model size (small, medium, large-v3-turbo)
  - int8 CPU quantization for 16GB RAM machines
  - Silero VAD preprocessing to reduce hallucinations
  - Per-word language detection using Unicode analysis + segment language
  - Word-level timestamps with confidence scores
  - Never forces uncertain non-English into English

Design principles:
  - This engine is an OBSERVER: it transcribes honestly, even if accuracy is low
  - Low-confidence words are flagged, not discarded or re-transcribed as English
  - Telugu/Hindi words detected by Unicode range or Whisper segment language
  - Code-switching within segments is preserved per-word
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from core.asr.base import (
    BaseASREngine, ASRWord, ASRSentence, ASRTranscript,
    ASRAlternative, detect_script, is_filler_word,
)
from core.logger import get_logger

logger = get_logger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

_SILENCE_MIN_MS = 200
_SENTENCE_SPLIT_MS = 400
_LOW_CONFIDENCE_THRESHOLD = 0.70
_ALTERNATE_HYPOTHESIS_THRESHOLD = 0.70  # Provide alternates when below this

# Languages we handle natively
_SUPPORTED_LANGUAGES = {"en", "te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "pa"}

# Telugu Romanized patterns (common words that Whisper might output in Latin script)
_TELUGU_ROMANIZED_PATTERNS = {
    "ante", "adi", "ala", "inka", "cheppanu", "telusa", "untadi",
    "chestham", "cheppandi", "manaku", "mana", "nenu", "meeru",
    "vaadu", "aayana", "aame", "evaru", "enti", "enduku", "epudu",
    "akkada", "ikkada", "chesthe", "cheppanu", "ayindi", "ledu",
    "unde", "kadhu", "avunu", "bagundi", "baguntundi", "choosthe",
    "raadhu", "vastundi", "vellu", "padutundi", "istam", "kashtam",
    "dhairyam", "badhulu", "prathi", "vishayam", "samasya", "janalaki",
    "okkasari", "chesthe", "lopala", "baita", "kani",
    "eppudu", "mari", "annadu", "annadi", "chesaru", "chesindi",
    "vachindi", "vellindi", "poyindi", "teesukunna", "cheppu",
}

# Common English words that appear in code-switched Telugu/Hindi speech.
# When the segment language is non-English but the word is clearly English,
# we tag the word as "en" to support code-switching detection.
# This is NOT an exhaustive English dictionary — just the most common words
# that appear in conversational Telugu/English code-switching.
_COMMON_ENGLISH_WORDS = {
    # Pronouns & articles
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "his", "her", "its", "our", "their", "the", "a", "an", "this", "that",
    # Verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "can", "may",
    "might", "must", "shall", "go", "come", "get", "make", "take", "give",
    "know", "think", "feel", "see", "hear", "say", "tell", "ask", "try",
    "need", "want", "help", "let", "keep", "find", "put", "mean", "become",
    "leave", "call", "show", "turn", "start", "run", "move", "live", "play",
    "believe", "hold", "bring", "happen", "write", "provide", "read", "stand",
    "lose", "pay", "meet", "include", "continue", "learn", "change", "watch",
    "follow", "stop", "create", "speak", "allow", "add", "grow", "open",
    "walk", "win", "offer", "remember", "love", "consider", "appear",
    "wait", "serve", "die", "send", "expect", "build", "stay", "fall",
    "cut", "reach", "kill", "remain", "suggest", "raise", "pass", "sell",
    "require", "report", "decide", "pull", "imagine", "understand",
    # Nouns
    "world", "life", "day", "time", "people", "way", "man", "woman", "child",
    "thing", "part", "place", "case", "week", "company", "system", "program",
    "question", "work", "government", "number", "night", "point", "home",
    "water", "room", "mother", "area", "money", "story", "fact", "month",
    "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "city", "community",
    "name", "president", "team", "minute", "idea", "body", "information",
    "back", "parent", "face", "moment", "girl", "boy", "opinion",
    "battle", "camera", "empathy", "judgment", "person", "birthday",
    "relationship", "emotion", "feeling", "broadcast", "secret",
    "patience", "anger", "happiness", "sadness", "problem", "solution",
    "situation", "experience", "behavior", "attitude", "perspective",
    "decision", "conversation", "communication", "reaction", "response",
    "opinion", "thought", "human", "society", "culture",
    # Adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next",
    "early", "young", "important", "few", "public", "bad", "same", "able",
    "worst", "best", "better", "worse", "late", "real", "true", "sure",
    "hard", "strong", "possible", "free", "clear", "simple", "personal",
    "professional", "social", "emotional",
    # Adverbs
    "not", "also", "very", "often", "however", "too", "usually", "really",
    "already", "always", "never", "sometimes", "together", "enough", "almost",
    "still", "ever", "just", "only", "even", "maybe", "probably",
    # Prepositions & conjunctions
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "up", "down", "but", "and", "or", "if", "when",
    "because", "while", "where", "how", "what", "which", "who", "why",
    # Common phrases in code-switching
    "don't", "can't", "won't", "isn't", "aren't", "doesn't", "didn't",
    "shouldn't", "wouldn't", "couldn't", "haven't", "hasn't",
    "no", "yes", "here", "there", "now", "then", "today", "tomorrow",
    "yesterday", "everyone", "everything", "someone", "something",
    "nothing", "anyone", "anything", "without",
}


# ─── Engine Implementation ──────────────────────────────────────────────────

class WhisperASREngine(BaseASREngine):
    """
    Multilingual Whisper ASR engine using faster-whisper (CTranslate2).

    Supports:
    - Multiple model sizes: small, medium, large-v3-turbo
    - CPU int8 quantization (fits in 16GB RAM)
    - Silero VAD for hallucination reduction
    - Per-word language tagging via Unicode analysis + segment language
    - Telugu/Hindi/English code-switching preservation
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[str] = None,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._model = None

    @property
    def engine_name(self) -> str:
        return f"faster-whisper-{self._model_size}"

    def is_available(self) -> bool:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            f"WhisperASR | Loading model '{self._model_size}' "
            f"(device={self._device}, compute_type={self._compute_type})..."
        )
        try:
            # First try loading from local HuggingFace cache without network call
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
                local_files_only=True,
            )
        except Exception:
            # If not in local cache or local load fails, allow download
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
                local_files_only=False,
            )
        logger.info(f"WhisperASR | Model '{self._model_size}' loaded successfully")

    def transcribe(
        self,
        audio_path: Path,
        language_hint: Optional[str] = None,
    ) -> ASRTranscript:
        """
        Transcribe audio with multilingual support.

        CRITICAL RULE: Never force uncertain Telugu/Hindi into English.
        If Whisper detects 'te' with confidence 0.42, we output language='te'
        with confidence=0.42 — we do NOT re-transcribe as English.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._ensure_model()

        t0 = time.time()

        # Track RAM usage
        try:
            import psutil
            process = psutil.Process()
            ram_before = process.memory_info().rss / (1024 * 1024)
        except Exception:
            ram_before = 0.0

        # ── Determine Whisper language parameter ──────────────────────
        # None = auto-detect (best for multilingual/code-switched content)
        # We only force a language if the user explicitly provides "en"
        whisper_language = None
        if language_hint == "en":
            whisper_language = "en"
        # For Telugu/Hindi/mixed: always auto-detect (Whisper handles it better)

        logger.info(
            f"WhisperASR | Transcribing '{audio_path.name}' "
            f"(hint={language_hint!r}, whisper_lang={whisper_language!r}, vad=True)"
        )

        # ── Run transcription ────────────────────────────────────────
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": _SILENCE_MIN_MS},
            language=whisper_language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        detected_language = info.language
        language_probability = round(info.language_probability, 3)

        logger.info(
            f"WhisperASR | Detected language: {detected_language!r} "
            f"(probability={language_probability:.3f})"
        )

        # ── CRITICAL: Do NOT re-transcribe if low confidence on non-English ──
        # The old code did:
        #   if prob < 0.6 and lang not in ("en", "hi"): re-transcribe with lang="en"
        # This DESTROYED Telugu. We explicitly do NOT do this.
        #
        # Instead, we flag low confidence and let downstream consumers decide.

        if language_probability < 0.5:
            logger.warning(
                f"WhisperASR | Low language confidence ({language_probability:.3f}) "
                f"for '{detected_language}'. Transcript may contain errors. "
                f"NOT re-transcribing — preserving original detection."
            )

        # Materialize segments
        segments = list(segments_gen)
        logger.info(f"WhisperASR | {len(segments)} segments transcribed")

        # ── Build word list with per-word language tagging ────────────
        all_words = self._build_word_list(segments, detected_language)
        logger.info(f"WhisperASR | {len(all_words)} words extracted")

        # ── Detect all languages present ─────────────────────────────
        languages_found = set()
        for w in all_words:
            languages_found.add(w.language)
        detected_languages = sorted(languages_found)

        # ── Build sentence list ──────────────────────────────────────
        sentences = self._build_sentences(all_words)
        logger.info(f"WhisperASR | {len(sentences)} sentences identified")

        # ── Build silence regions ────────────────────────────────────
        silence_regions = self._find_silence_regions(all_words)
        logger.info(f"WhisperASR | {len(silence_regions)} silence regions detected")

        # ── Compute speech stats ─────────────────────────────────────
        speech_stats = self._compute_speech_stats(all_words, sentences, silence_regions)

        # ── Measure RAM ──────────────────────────────────────────────
        try:
            ram_after = process.memory_info().rss / (1024 * 1024)
            peak_ram = round(ram_after - ram_before, 1)
        except Exception:
            peak_ram = 0.0

        transcription_time = round(time.time() - t0, 2)

        full_text = " ".join(w.word for w in all_words).strip()

        transcript = ASRTranscript(
            full_text=full_text,
            detected_languages=detected_languages,
            primary_language=detected_language,
            language_confidence=language_probability,
            whisper_model=self._model_size,
            words=all_words,
            sentences=sentences,
            silence_regions=silence_regions,
            speech_stats=speech_stats,
            engine_name=self.engine_name,
            compute_type=self._compute_type,
            device=self._device,
            transcription_time_seconds=transcription_time,
            peak_ram_mb=peak_ram,
        )

        logger.info(
            f"WhisperASR | Transcription complete in {transcription_time}s. "
            f"Languages: {detected_languages}, Words: {len(all_words)}, "
            f"Sentences: {len(sentences)}, RAM delta: {peak_ram:.0f}MB"
        )

        return transcript

    # ─── Word-Level Building ─────────────────────────────────────────────────

    def _build_word_list(
        self,
        segments: list,
        detected_language: str,
    ) -> List[ASRWord]:
        """
        Extract a flat list of ASRWord objects with per-word language tagging.

        Language tagging strategy:
        1. If Whisper segment language is set, use it as the base
        2. Check Unicode: Telugu glyphs → language="te"
        3. Check romanized Telugu patterns → language="te"
        4. Otherwise use the segment/detected language
        """
        words: List[ASRWord] = []
        word_id = 0

        for segment in segments:
            if not segment.words:
                continue

            # Segment-level language (if available)
            seg_lang = getattr(segment, "language", detected_language) or detected_language

            for w in segment.words:
                word_text = w.word.strip()
                if not word_text:
                    continue

                confidence = round(w.probability, 3)

                # ── Per-word language detection ───────────────────────
                word_lang = self._detect_word_language(word_text, seg_lang)
                word_script = detect_script(word_text)

                # Romanized form for non-Latin scripts
                romanized = None
                if word_script != "latin":
                    romanized = word_text  # Already in original script

                # Alternate hypotheses for low-confidence words
                alternates: List[ASRAlternative] = []
                if confidence < _ALTERNATE_HYPOTHESIS_THRESHOLD:
                    # We don't have actual alternates from faster-whisper,
                    # but we flag the uncertainty
                    alternates = []

                asr_word = ASRWord(
                    id=word_id,
                    word=word_text,
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    duration=round(w.end - w.start, 3),
                    confidence=confidence,
                    language=word_lang,
                    script=word_script,
                    romanized=romanized,
                    is_filler=is_filler_word(word_text),
                    low_confidence_flag=confidence < _LOW_CONFIDENCE_THRESHOLD,
                    alternate_hypotheses=alternates,
                )
                words.append(asr_word)
                word_id += 1

        return words

    def _detect_word_language(self, word_text: str, segment_language: str) -> str:
        """
        Detect the language of a single word.

        Priority:
        1. Telugu Unicode glyphs -> "te"
        2. Devanagari Unicode glyphs -> "hi"
        3. Latin script + common English word -> "en" (even in non-English segments)
        4. Latin script + romanized Telugu pattern -> "te"
        5. Segment-level language from Whisper (fallback)

        This is critical for code-switching: when Whisper detects the segment
        as Telugu but the word is clearly English (e.g. "empathy", "world"),
        we tag it as "en" to support accurate language_mix detection.
        """
        # Check Unicode ranges first (most reliable)
        script = detect_script(word_text)
        if script == "telugu":
            return "te"
        if script == "devanagari":
            return "hi"

        # For Latin-script words, determine if English or romanized Indic
        cleaned = word_text.lower().strip(".,!?;:\"'")

        # Check romanized Telugu patterns first (higher specificity)
        if cleaned in _TELUGU_ROMANIZED_PATTERNS:
            return "te"

        # Check if it's a common English word
        # This is especially important when segment_language != "en"
        if cleaned in _COMMON_ENGLISH_WORDS:
            return "en"

        # For longer Latin words (3+ chars) in non-English segments,
        # heuristically check if they look English
        if segment_language != "en" and len(cleaned) >= 3:
            # Words with common English suffixes are likely English
            english_suffixes = (
                "tion", "sion", "ment", "ness", "able", "ible", "ful",
                "less", "ous", "ive", "ing", "tion", "ally", "ity",
                "ence", "ance", "ory", "ary",
            )
            if any(cleaned.endswith(s) for s in english_suffixes):
                return "en"

        # Fall back to segment language
        return segment_language

    # ─── Sentence Building ───────────────────────────────────────────────────

    def _build_sentences(self, all_words: List[ASRWord]) -> List[ASRSentence]:
        """
        Group words into sentences using punctuation boundaries.
        Each sentence carries language_mix metadata.
        """
        if not all_words:
            return []

        sentences: List[ASRSentence] = []
        sentence_id = 0
        current_words: List[ASRWord] = []
        sentence_end_re = re.compile(r'[.!?…]$')

        def _flush(word_list: List[ASRWord]) -> None:
            nonlocal sentence_id
            if not word_list:
                return

            text = " ".join(w.word for w in word_list).strip()
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)

            # Determine language mix
            langs_in_sentence = sorted(set(w.language for w in word_list))

            filler_count = sum(1 for w in word_list if w.is_filler)
            low_conf_count = sum(1 for w in word_list if w.low_confidence_flag)

            sentences.append(ASRSentence(
                id=sentence_id,
                text=text,
                start=word_list[0].start,
                end=word_list[-1].end,
                duration=round(word_list[-1].end - word_list[0].start, 3),
                word_count=len(word_list),
                word_ids=[w.id for w in word_list],
                language_mix=langs_in_sentence,
                filler_count=filler_count,
                has_filler=filler_count > 0,
                low_confidence_flag=low_conf_count > len(word_list) * 0.3,
            ))
            sentence_id += 1

        for word in all_words:
            current_words.append(word)
            if sentence_end_re.search(word.word):
                _flush(current_words)
                current_words = []

        _flush(current_words)  # Remaining words
        return sentences

    # ─── Silence Region Detection ────────────────────────────────────────────

    def _find_silence_regions(self, all_words: List[ASRWord]) -> List[Dict[str, Any]]:
        """Find gaps between consecutive words longer than _SILENCE_MIN_MS."""
        if len(all_words) < 2:
            return []

        silence_regions: List[Dict[str, Any]] = []
        sentence_end_re = re.compile(r'[.!?…]$')

        for i in range(len(all_words) - 1):
            current = all_words[i]
            next_word = all_words[i + 1]
            gap = next_word.start - current.end

            if gap * 1000 < _SILENCE_MIN_MS:
                continue

            is_boundary = bool(sentence_end_re.search(current.word))
            position = "BETWEEN_SENTENCES" if is_boundary else "MID_SENTENCE"

            silence_regions.append({
                "id": len(silence_regions),
                "start": round(current.end, 3),
                "end": round(next_word.start, 3),
                "duration_seconds": round(gap, 3),
                "duration_ms": round(gap * 1000, 0),
                "position": position,
                "word_before_id": current.id,
                "word_before_text": current.word,
                "word_after_id": next_word.id,
                "word_after_text": next_word.word,
                # Populated by Stage 3/5:
                "preceding_energy": None,
                "following_energy": None,
                "classification": "PENDING",
                "preserve_recommendation": None,
                "preserve_reason": None,
            })

        return silence_regions

    # ─── Speech Statistics ───────────────────────────────────────────────────

    def _compute_speech_stats(
        self,
        all_words: List[ASRWord],
        sentences: List[ASRSentence],
        silence_regions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute aggregate speech statistics."""
        if not all_words:
            return {}

        total_duration = all_words[-1].end - all_words[0].start
        speech_seconds = sum(w.duration for w in all_words)
        silence_seconds = sum(s["duration_seconds"] for s in silence_regions)
        filler_count = sum(1 for w in all_words if w.is_filler)
        low_conf_count = sum(1 for w in all_words if w.low_confidence_flag)

        # Language distribution
        lang_counts: Dict[str, int] = {}
        for w in all_words:
            lang_counts[w.language] = lang_counts.get(w.language, 0) + 1

        # WPM computation
        avg_wpm = self._compute_wpm(sentences)

        return {
            "total_words": len(all_words),
            "total_sentences": len(sentences),
            "speech_duration_seconds": round(speech_seconds, 2),
            "silence_duration_seconds": round(silence_seconds, 2),
            "speech_ratio": round(speech_seconds / max(total_duration, 1), 3),
            "average_wpm": avg_wpm,
            "filler_word_count": filler_count,
            "filler_percentage": round(100 * filler_count / max(len(all_words), 1), 1),
            "low_confidence_word_count": low_conf_count,
            "low_confidence_percentage": round(100 * low_conf_count / max(len(all_words), 1), 1),
            "language_distribution": lang_counts,
            "code_switching_detected": len(lang_counts) > 1,
        }

    def _compute_wpm(self, sentences: List[ASRSentence]) -> float:
        """Compute average speaking rate in words per minute."""
        valid = [s for s in sentences if s.duration > 0.5 and s.word_count >= 3]
        if not valid:
            return 0.0
        rates = [(s.word_count / s.duration) * 60 for s in valid]
        return round(sum(rates) / len(rates), 1)
