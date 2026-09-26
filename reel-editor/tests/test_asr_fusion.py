"""
tests/test_asr_fusion.py

Unit tests for the Evidence-Based Multi-ASR Fusion Engine.
Validates:
  1. Multi-factor candidate scoring (confidence, loanword detection, context continuity)
  2. Non-naive selection: IndicConformer does NOT blindly override Whisper
  3. Candidate audit trail: preserves competing hypotheses with selection_reason
  4. Script vs spoken language separation (language='te', script='latin', romanized=...)
  5. English loanword retention inside Telugu context
"""

import pytest
from core.asr.base import ASRWord, ASRCandidate
from core.asr.fusion import ASRFusionEngine
from core.asr.transliterator import telugu_word_to_roman, is_english_loanword


class TestASRFusionEngine:
    def setup_method(self):
        self.engine = ASRFusionEngine(time_tolerance_s=0.35)

    def test_candidate_audit_trail_preserved(self):
        """Audit trail must contain all competing hypotheses, selection, and a concise reason."""
        w_whisper = ASRWord(
            id=0, word="everyday", start=8.42, end=8.91, duration=0.49,
            confidence=0.88, language="en", script="latin", source_engine="whisper"
        )
        w_indic = ASRWord(
            id=0, word="ఎవ్రీడే", start=8.40, end=8.95, duration=0.55,
            confidence=0.72, language="te", script="telugu", romanized_text="everyday",
            native_script_text="ఎవ్రీడే", source_engine="indic_conformer"
        )

        fused_words = self.engine.fuse_words([w_whisper], [w_indic])
        assert len(fused_words) == 1
        fused = fused_words[0]

        # Audit requirements:
        assert len(fused.candidates) == 2
        candidate_engines = {c.engine for c in fused.candidates}
        assert "whisper" in candidate_engines
        assert "indic_conformer" in candidate_engines
        assert fused.selection_reason != ""
        assert fused.source_engine in ("whisper", "indic_conformer", "fused", "consensus")

    def test_indic_conformer_does_not_blindly_win(self):
        """When Whisper has high confidence on clear English, IndicConformer should not blindly win."""
        w_whisper = ASRWord(
            id=0, word="empathy", start=2.30, end=2.95, duration=0.65,
            confidence=0.96, language="en", script="latin", source_engine="whisper"
        )
        # Suppose IndicConformer hallucinates a noisy Telugu phonetic match
        w_indic = ASRWord(
            id=0, word="ఏంపతి", start=2.25, end=2.98, duration=0.73,
            confidence=0.60, language="te", script="telugu", romanized_text="empathi",
            source_engine="indic_conformer"
        )

        fused = self.engine.fuse_words([w_whisper], [w_indic])[0]
        assert fused.word == "empathy"
        assert fused.language == "en"
        assert fused.source_engine in ("whisper", "fused")

    def test_indic_conformer_wins_on_genuine_telugu_word(self):
        """When Whisper garbles Telugu speech and IndicConformer is confident, IndicConformer wins."""
        # Whisper garbles 'Kani' as 'Can he' or 'can' with low confidence
        w_whisper = ASRWord(
            id=0, word="can", start=8.55, end=8.90, duration=0.35,
            confidence=0.38, language="en", script="latin", source_engine="whisper"
        )
        w_indic = ASRWord(
            id=0, word="కానీ", start=8.50, end=8.92, duration=0.42,
            confidence=0.91, language="te", script="telugu", romanized_text="kani",
            native_script_text="కానీ", source_engine="indic_conformer"
        )

        fused = self.engine.fuse_words([w_whisper], [w_indic])[0]
        assert fused.language == "te"
        assert fused.source_engine == "indic_conformer"
        assert fused.romanized_text == "kani"
        assert fused.native_script_text == "కానీ"

    def test_english_loanword_inside_telugu_context(self):
        """English loanwords like 'observe', 'life' spoken in Telugu should retain English representation."""
        w_whisper = ASRWord(
            id=0, word="observe", start=10.95, end=11.55, duration=0.60,
            confidence=0.85, language="en", script="latin", source_engine="whisper"
        )
        w_indic = ASRWord(
            id=0, word="అబ్సర్వ్", start=10.90, end=11.50, duration=0.60,
            confidence=0.88, language="te", script="telugu", romanized_text="observe",
            native_script_text="అబ్సర్వ్", source_engine="indic_conformer"
        )

        fused = self.engine.fuse_words([w_whisper], [w_indic])[0]
        assert fused.word.lower() == "observe"
        assert fused.language == "en"

    def test_script_vs_spoken_language_separation(self):
        """Spoken Telugu with Romanized Latin text must have language=te, script=latin, romanized=..."""
        w_indic = ASRWord(
            id=0, word="ఒక్కసారి", start=10.30, end=10.95, duration=0.65,
            confidence=0.92, language="te", script="telugu", romanized_text="okkasari",
            native_script_text="ఒక్కసారి", source_engine="indic_conformer"
        )
        # Empty whisper candidate for this slot
        fused = self.engine.fuse_words([], [w_indic])[0]
        assert fused.language == "te"
        assert fused.script == "latin"
        assert fused.romanized_text == "okkasari"
        assert fused.native_script_text == "ఒక్కసారి"
