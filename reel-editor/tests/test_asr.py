"""
tests/test_asr.py

Unit Tests for the Professional ASR Subsystem
──────────────────────────────────────────────
Tests:
  1. ASRManager initialization and availability
  2. ASRWord/ASRSentence/ASRTranscript data contracts
  3. Per-word language detection (Telugu Unicode, romanized patterns)
  4. Filler word detection (English + Telugu)
  5. Script detection (Latin, Telugu, Devanagari)
  6. Sentence building with language_mix
  7. Integration: full transcription of Surya.mp4 audio (if available)

Run: python -m pytest tests/test_asr.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from core.asr.base import (
    ASRWord, ASRSentence, ASRTranscript, ASRAlternative,
    detect_script, is_filler_word,
)
from core.asr.whisper_engine import WhisperASREngine
from core.asr.manager import ASRManager


# ─── Test: Data Contract Serialization ───────────────────────────────────────

class TestASRWordContract:
    """Test ASRWord data contract serialization and deserialization."""

    def test_basic_english_word(self):
        word = ASRWord(
            id=0, word="empathy", start=0.18, end=0.78,
            duration=0.6, confidence=0.92, language="en",
            script="latin", is_filler=False,
        )
        d = word.to_dict()
        assert d["word"] == "empathy"
        assert d["language"] == "en"
        assert d["script"] == "latin"
        assert d["confidence"] == 0.92
        assert d["romanized"] is None

    def test_telugu_unicode_word(self):
        word = ASRWord(
            id=1, word="తెలుసా", start=1.0, end=1.5,
            duration=0.5, confidence=0.75, language="te",
            script="telugu", romanized="telusa",
        )
        d = word.to_dict()
        assert d["language"] == "te"
        assert d["script"] == "telugu"
        assert d["romanized"] == "telusa"

    def test_low_confidence_with_alternates(self):
        word = ASRWord(
            id=2, word="cheppanu", start=2.0, end=2.5,
            duration=0.5, confidence=0.45, language="te",
            script="latin", romanized="cheppanu",
            low_confidence_flag=True,
            alternate_hypotheses=[
                ASRAlternative(word="chepanu", confidence=0.12, language="te"),
            ],
        )
        d = word.to_dict()
        assert d["low_confidence_flag"] is True
        assert len(d["alternate_hypotheses"]) == 1
        assert d["alternate_hypotheses"][0]["word"] == "chepanu"

    def test_roundtrip_serialization(self):
        word = ASRWord(
            id=5, word="judgment", start=3.0, end=3.4,
            duration=0.4, confidence=0.88, language="en",
            script="latin", is_filler=False,
            alternate_hypotheses=[
                ASRAlternative(word="judgement", confidence=0.08, language="en"),
            ],
        )
        d = word.to_dict()
        restored = ASRWord.from_dict(d)
        assert restored.word == "judgment"
        assert restored.language == "en"
        assert len(restored.alternate_hypotheses) == 1


class TestASRSentenceContract:
    """Test ASRSentence data contract."""

    def test_code_switched_sentence(self):
        sentence = ASRSentence(
            id=0,
            text="Kani mana everyday life lo",
            start=0.0, end=3.0, duration=3.0,
            word_count=5,
            word_ids=[0, 1, 2, 3, 4],
            language_mix=["en", "te"],
            filler_count=0, has_filler=False,
        )
        d = sentence.to_dict()
        assert d["language_mix"] == ["en", "te"]
        assert d["word_count"] == 5

    def test_pure_english_sentence(self):
        sentence = ASRSentence(
            id=1,
            text="Having a worst day of your life.",
            start=3.0, end=6.0, duration=3.0,
            word_count=7,
            word_ids=[5, 6, 7, 8, 9, 10, 11],
            language_mix=["en"],
        )
        assert sentence.language_mix == ["en"]


class TestASRTranscriptContract:
    """Test ASRTranscript data contract."""

    def test_full_transcript_serialization(self):
        transcript = ASRTranscript(
            full_text="Empathy ante telusa.",
            detected_languages=["en", "te"],
            primary_language="te",
            language_confidence=0.72,
            whisper_model="small",
            words=[
                ASRWord(id=0, word="Empathy", start=0.0, end=0.5, duration=0.5,
                        confidence=0.9, language="en", script="latin"),
                ASRWord(id=1, word="ante", start=0.5, end=0.8, duration=0.3,
                        confidence=0.7, language="te", script="latin", romanized="ante"),
                ASRWord(id=2, word="telusa.", start=0.8, end=1.2, duration=0.4,
                        confidence=0.65, language="te", script="latin", romanized="telusa"),
            ],
            sentences=[
                ASRSentence(id=0, text="Empathy ante telusa.", start=0.0, end=1.2,
                           duration=1.2, word_count=3, word_ids=[0, 1, 2],
                           language_mix=["en", "te"]),
            ],
        )
        d = transcript.to_dict()
        assert d["detected_languages"] == ["en", "te"]
        assert d["primary_language"] == "te"
        assert len(d["words"]) == 3
        assert d["words"][1]["language"] == "te"


# ─── Test: Script Detection ─────────────────────────────────────────────────

class TestScriptDetection:
    """Test Unicode-based script detection."""

    def test_latin_text(self):
        assert detect_script("empathy") == "latin"
        assert detect_script("Hello World") == "latin"

    def test_telugu_text(self):
        assert detect_script("తెలుసా") == "telugu"
        assert detect_script("ఎంపతి") == "telugu"

    def test_devanagari_text(self):
        assert detect_script("नमस्ते") == "devanagari"
        assert detect_script("हिंदी") == "devanagari"

    def test_mixed_defaults_to_script_found(self):
        # If Telugu chars present, detect as Telugu
        assert detect_script("Hello తెలుసా") == "telugu"

    def test_empty_string(self):
        assert detect_script("") == "latin"

    def test_numbers_and_punctuation(self):
        assert detect_script("123!@#") == "latin"


# ─── Test: Filler Word Detection ────────────────────────────────────────────

class TestFillerDetection:
    """Test English and Telugu filler word detection."""

    def test_english_fillers(self):
        assert is_filler_word("um") is True
        assert is_filler_word("Uh") is True
        assert is_filler_word("like") is True
        assert is_filler_word("basically,") is True
        assert is_filler_word("you know") is True

    def test_telugu_fillers(self):
        assert is_filler_word("matlab") is True
        assert is_filler_word("yaar") is True
        assert is_filler_word("ante") is True

    def test_non_fillers(self):
        assert is_filler_word("empathy") is False
        assert is_filler_word("judgment") is False
        assert is_filler_word("important") is False

    def test_punctuation_stripping(self):
        assert is_filler_word("um,") is True
        assert is_filler_word("like.") is True
        assert is_filler_word("basically!") is True


# ─── Test: Language Detection ────────────────────────────────────────────────

class TestLanguageDetection:
    """Test per-word language detection in WhisperASREngine."""

    @pytest.fixture
    def engine(self):
        return WhisperASREngine(model_size="small", device="cpu", compute_type="int8")

    def test_telugu_unicode_word(self, engine):
        lang = engine._detect_word_language("తెలుసా", "en")
        assert lang == "te"

    def test_devanagari_word(self, engine):
        lang = engine._detect_word_language("नमस्ते", "en")
        assert lang == "hi"

    def test_romanized_telugu_word(self, engine):
        lang = engine._detect_word_language("cheppanu", "en")
        assert lang == "te"

    def test_romanized_telugu_mana(self, engine):
        lang = engine._detect_word_language("mana", "en")
        assert lang == "te"

    def test_english_word_with_english_segment(self, engine):
        lang = engine._detect_word_language("empathy", "en")
        assert lang == "en"

    def test_english_word_with_telugu_segment(self, engine):
        # An English word in a Telugu segment should be detected as "en"
        # because of our code-switching english dictionary detection
        lang = engine._detect_word_language("birthday", "te")
        assert lang == "en"  # Detected as English word in code-switching

    def test_unknown_word_uses_segment_language(self, engine):
        # An unknown Latin-script word not in English or Romanized Telugu
        # falls back to the segment language
        lang = engine._detect_word_language("xyzabc", "te")
        assert lang == "te"


# ─── Test: ASRManager Initialization ─────────────────────────────────────────

class TestASRManager:
    """Test ASRManager initialization and availability."""

    def test_initialization(self):
        manager = ASRManager(model_size="small", device="cpu", compute_type="int8")
        assert manager.engine_name == "faster-whisper-small"

    def test_is_available(self):
        manager = ASRManager(model_size="small")
        # Should be True if faster-whisper is installed
        available = manager.is_available()
        assert isinstance(available, bool)


# ─── Test: Integration (requires Surya.mp4 and model download) ──────────────

class TestIntegration:
    """
    Integration tests that require actual audio files and model downloads.
    These are slow and are marked with @pytest.mark.slow.
    Skip them with: pytest -m "not slow"
    """

    @pytest.fixture
    def surya_audio(self):
        """Check if Surya.mp4 audio exists."""
        # Stage 1 extracts audio.wav to outputs/
        candidates = [
            PROJECT_ROOT / "outputs" / "Surya" / "audio.wav",
            PROJECT_ROOT / "outputs" / "audio.wav",
        ]
        for p in candidates:
            if p.exists():
                return p
        pytest.skip("Surya.mp4 audio not found — run Stage 1 first")

    @pytest.mark.slow
    def test_full_transcription(self, surya_audio, tmp_path):
        """Full transcription of Surya.mp4 audio with quality checks."""
        manager = ASRManager(model_size="small", device="cpu", compute_type="int8")
        transcript = manager.transcribe(surya_audio, language_hint=None)

        # Basic checks
        assert len(transcript.words) > 0, "No words transcribed"
        assert len(transcript.sentences) > 0, "No sentences formed"
        assert transcript.primary_language in {"en", "te", "hi"}, \
            f"Unexpected primary language: {transcript.primary_language}"

        # Every word must have a language tag
        for w in transcript.words:
            assert w.language in {"en", "te", "hi", "mixed"}, \
                f"Word '{w.word}' has invalid language: {w.language}"
            assert w.script in {"latin", "telugu", "devanagari"}, \
                f"Word '{w.word}' has invalid script: {w.script}"

        # Every sentence must have language_mix
        for s in transcript.sentences:
            assert isinstance(s.language_mix, list), \
                f"Sentence {s.id} missing language_mix"
            assert len(s.language_mix) > 0

        # Save for inspection
        out_path = tmp_path / "transcript_raw.json"
        from core.json_utils import save_json
        save_json(transcript.to_dict(), out_path)

        # Print quality report
        print(f"\n{'='*60}")
        print(f"INTEGRATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Words:        {len(transcript.words)}")
        print(f"Sentences:    {len(transcript.sentences)}")
        print(f"Languages:    {transcript.detected_languages}")
        print(f"Primary:      {transcript.primary_language} ({transcript.language_confidence:.3f})")
        print(f"Code-switch:  {transcript.speech_stats.get('code_switching_detected')}")
        print(f"Low conf:     {transcript.speech_stats.get('low_confidence_percentage', 0):.1f}%")
        print(f"Time:         {transcript.transcription_time_seconds}s")
        print(f"RAM delta:    {transcript.peak_ram_mb:.0f} MB")
        print(f"Preview:      {transcript.full_text[:200]}")
        print(f"{'='*60}")

    @pytest.mark.slow
    def test_telugu_not_destroyed(self, surya_audio):
        """
        CRITICAL TEST: Verify Telugu words are NOT overwritten with English.

        The old ASR had a hardcoded fallback that re-transcribed uncertain Telugu
        as English, turning 'Kani mana everyday life lo' into
        'Can't even one eye every day life no cuss'.

        This test verifies that bug is fixed.
        """
        manager = ASRManager(model_size="small", device="cpu", compute_type="int8")
        transcript = manager.transcribe(surya_audio, language_hint=None)

        # The transcript should NOT be 100% English if the source has Telugu
        lang_dist = transcript.speech_stats.get("language_distribution", {})

        print(f"\nLanguage Distribution: {lang_dist}")
        print(f"Detected Languages: {transcript.detected_languages}")

        # If Surya.mp4 contains Telugu, we should see "te" in detected languages
        # or at minimum the system should NOT have forced everything to English
        assert transcript.primary_language != "en" or len(transcript.detected_languages) > 1 or \
            transcript.language_confidence < 0.7, \
            "Expected multilingual detection but got pure English with high confidence. " \
            "Telugu may have been destroyed by English fallback."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
