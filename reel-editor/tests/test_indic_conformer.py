"""
tests/test_indic_conformer.py

Unit tests for AI4Bharat IndicConformer Telugu integration and transliterator.
Validates:
  1. Telugu Unicode to Latin phonetic transliteration
  2. Loanword detection for common conversational English vocabulary
  3. Word segmentation and timestamp assignment
  4. Script vs language attribution
"""

import pytest
from core.asr.transliterator import (
    telugu_word_to_roman,
    transliterate_text,
    is_english_loanword,
    detect_script_extended,
)


class TestTeluguTransliterator:
    def test_telugu_characters_to_roman(self):
        # Test core Telugu phonetic mappings
        assert telugu_word_to_roman("కానీ") == "kani"
        assert telugu_word_to_roman("మన") == "mana"
        assert telugu_word_to_roman("ఒక్కసారి") == "okkasari"
        assert telugu_word_to_roman("చేస్తే") == "chesthe"
        assert telugu_word_to_roman("అర్థమవుతుంది") == "arthamavuthundi"

    def test_english_loanwords_phonetic_normalization(self):
        # Words borrowed from English phonetically transcribed in Telugu
        assert telugu_word_to_roman("ఎవ్రీడే") == "everyday"
        assert telugu_word_to_roman("లైఫ్") == "life"
        assert telugu_word_to_roman("అబ్సర్వ్") == "observe"
        assert telugu_word_to_roman("వర్డ్") == "word"
        assert telugu_word_to_roman("వేల్యూ") == "value"

    def test_transliterate_sentence(self):
        telugu_sent = "కానీ మన ఎవ్రీడే లైఫ్ లో ఒక్కసారి అబ్సర్వ్ చేస్తే"
        romanized = transliterate_text(telugu_sent)
        assert "kani" in romanized
        assert "mana" in romanized
        assert "everyday" in romanized
        assert "observe" in romanized

    def test_is_english_loanword(self):
        assert is_english_loanword("everyday") is True
        assert is_english_loanword("observe") is True
        assert is_english_loanword("life") is True
        assert is_english_loanword("podcast") is True
        assert is_english_loanword("okkasari") is False
        assert is_english_loanword("chesthe") is False

    def test_detect_script_extended(self):
        assert detect_script_extended("కానీ") == "telugu"
        assert detect_script_extended("everyday") == "latin"
        assert detect_script_extended("observe చేస్తే") == "mixed"
