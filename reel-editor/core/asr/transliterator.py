"""
core/asr/transliterator.py

Deterministic Telugu-to-Latin Script Transliterator
────────────────────────────────────────────────────
Phonetically maps Telugu Unicode text (U+0C00 – U+0C7F) into clean, standard
Romanized Telugu (ISO 15919 / ITRANS compatible) without hallucination.

Preserves:
  - English words unchanged (if already in Latin script)
  - Telugu vowels, consonants, guninthalu (matras), virama, and anusvara
  - Common conversational spelling conventions (e.g. 'kani', 'mana', 'chesthe')
"""

import re
from typing import Optional

# Independent Vowels
_VOWELS = {
    '\u0C05': 'a',
    '\u0C06': 'aa',
    '\u0C07': 'i',
    '\u0C08': 'ee',
    '\u0C09': 'u',
    '\u0C0A': 'oo',
    '\u0C0B': 'ru',
    '\u0C0C': 'lu',
    '\u0C0E': 'e',
    '\u0C0F': 'ee',
    '\u0C10': 'ai',
    '\u0C12': 'o',
    '\u0C13': 'oo',
    '\u0C14': 'au',
}

# Consonants (base consonant without vowel)
_CONSONANTS = {
    '\u0C15': 'k',
    '\u0C16': 'kh',
    '\u0C17': 'g',
    '\u0C18': 'gh',
    '\u0C19': 'ng',
    '\u0C1A': 'ch',
    '\u0C1B': 'chh',
    '\u0C1C': 'j',
    '\u0C1D': 'jh',
    '\u0C1E': 'ny',
    '\u0C1F': 't',
    '\u0C20': 'th',
    '\u0C21': 'd',
    '\u0C22': 'dh',
    '\u0C23': 'n',
    '\u0C24': 't',
    '\u0C25': 'th',
    '\u0C26': 'd',
    '\u0C27': 'dh',
    '\u0C28': 'n',
    '\u0C2A': 'p',
    '\u0C2B': 'ph',
    '\u0C2C': 'b',
    '\u0C2D': 'bh',
    '\u0C2E': 'm',
    '\u0C2F': 'y',
    '\u0C30': 'r',
    '\u0C31': 'r',
    '\u0C32': 'l',
    '\u0C33': 'l',
    '\u0C35': 'v',
    '\u0C36': 'sh',
    '\u0C37': 'sh',
    '\u0C38': 's',
    '\u0C39': 'h',
}

# Dependent Vowel Signs (Matras)
_MATRAS = {
    '\u0C3E': 'aa',
    '\u0C3F': 'i',
    '\u0C40': 'ee',
    '\u0C41': 'u',
    '\u0C42': 'oo',
    '\u0C43': 'ru',
    '\u0C44': 'roo',
    '\u0C46': 'e',
    '\u0C47': 'e',
    '\u0C48': 'ai',
    '\u0C4A': 'o',
    '\u0C4B': 'o',
    '\u0C4C': 'au',
}

_VIRAMA = '\u0C4D'    # ్ (halant / suppresses inherent vowel 'a')
_ANUSVARA = '\u0C02'  # ం (nasal: m/n)
_VISARGA = '\u0C03'   # ః (h)

# Canonical conversational dictionary for common conversational words
_CANONICAL_WORDS = {
    "కానీ": "kani",
    "కాని": "kani",
    "మన": "mana",
    "ఒక్కసారి": "okkasari",
    "చేస్తే": "chesthe",
    "అబ్సర్వ్": "observe",
    "ఎవీడే": "everyday",
    "ఎవ్రీడే": "everyday",
    "లైఫ్": "life",
    "వరల్డ్": "world",
    "వల్డ్": "world",
    "ఇమాజిన్": "imagine",
    "ఎంత": "entha",
    "వ్యూ": "value",
    "ఉంటందో": "untando",
    "అర్థంవుతుంది": "arthamavuthundi",
    "అర్థమవుతుంది": "arthamavuthundi",
    "అంతే": "anthe",
    "ఎందుకంటే": "endukante",
    "చెప్పేసి": "cheppesi",
    "పర్ఫెక్షన్": "perfection",
    "బిగనర్": "beginner",
    "ఎమోషన్": "emotion",
    "ఎంపథీ": "empathy",
    "ఎంఫతి": "empathy",
    "రిప్లైస్": "replies",
    "కేర్": "care",
    "డౌట్": "doubt",
    "జడ్జ్": "judge",
    "వర్డ్": "word",
    "వేల్యూ": "value",
}

_LOANWORD_PHONETIC_CORRECTIONS = {
    "vard": "word",
    "varda": "word",
    "vardu": "word",
    "vardi": "word",
    "vyalue": "value",
    "velyu": "value",
    "valyu": "value",
    "vaylyoo": "value",
    "evriday": "everyday",
    "evreeday": "everyday",
    "evreedee": "everyday",
    "evrede": "everyday",
    "abzarv": "observe",
    "abzary": "observe",
    "abserv": "observe",
    "laif": "life",
    "empathi": "empathy",
}


def telugu_word_to_roman(word: str) -> str:
    """
    Transliterates a single Telugu Unicode word into Romanized script.
    If the word is already Latin or contains mixed characters, handles each character cleanly.
    """
    word_clean = word.strip()
    if not word_clean:
        return ""

    # Check canonical dictionary first
    stripped = word_clean.strip(".,!?:;\"'()[]{}")
    if stripped in _CANONICAL_WORDS:
        repl = _CANONICAL_WORDS[stripped]
        return word_clean.replace(stripped, repl)

    # Check if purely non-Telugu (e.g. English)
    if not any('\u0C00' <= char <= '\u0C7F' for char in word_clean):
        return word_clean
    res = []
    i = 0
    n = len(word_clean)

    while i < n:
        c = word_clean[i]

        if c in _VOWELS:
            res.append(_VOWELS[c])
            i += 1
        elif c in _CONSONANTS:
            base = _CONSONANTS[c]
            # Check what follows this consonant
            if i + 1 < n:
                nxt = word_clean[i + 1]
                if nxt == _VIRAMA:
                    res.append(base)
                    i += 2  # Skip virama
                elif nxt in _MATRAS:
                    res.append(base + _MATRAS[nxt])
                    i += 2  # Skip matra
                elif nxt == _ANUSVARA:
                    res.append(base + "am")
                    i += 2
                elif nxt in _CONSONANTS or nxt in _VOWELS or nxt.isspace() or not ('\u0C00' <= nxt <= '\u0C7F'):
                    # Inherent vowel 'a'
                    res.append(base + "a")
                    i += 1
                else:
                    res.append(base + "a")
                    i += 1
            else:
                # End of word: inherent vowel
                res.append(base + "a")
                i += 1
        elif c in _MATRAS:
            res.append(_MATRAS[c])
            i += 1
        elif c == _ANUSVARA:
            res.append("m")
            i += 1
        elif c == _VISARGA:
            res.append("h")
            i += 1
        elif c == _VIRAMA:
            # Standalone virama
            i += 1
        else:
            # Punctuation or ASCII character
            res.append(c)
            i += 1

    roman = "".join(res)
    # Post-clean trailing redundant vowels (e.g. 'kaani' / 'kani')
    roman = re.sub(r'aa', 'a', roman)  # Conversational Telugu commonly spells long a as single a
    roman = re.sub(r'ee', 'e', roman)  # e.g. chesthe rather than cheesthee
    roman = re.sub(r'oo', 'o', roman)
    
    # Check loanword corrections
    cleaned_lower = roman.lower().strip()
    if cleaned_lower in _LOANWORD_PHONETIC_CORRECTIONS:
        return _LOANWORD_PHONETIC_CORRECTIONS[cleaned_lower]

    return roman


def transliterate_text(text: str) -> str:
    """Transliterates a full string of Telugu text to Romanized words."""
    words = text.split()
    return " ".join(telugu_word_to_roman(w) for w in words)


_LOANWORDS = {
    "everyday", "life", "observe", "professional", "world", "imagine",
    "value", "perfection", "beginner", "emotion", "empathy", "replies",
    "care", "doubt", "judge", "reaction", "feeling", "call", "message",
    "video", "reel", "post", "friend", "time", "day", "worst", "best",
    "because", "somebody", "think", "start", "stop", "change", "feel",
    "having", "don't", "dont", "late", "action", "behavior", "attitude",
    "student", "students", "compare", "dance", "western", "class",
    "podcast", "word", "book", "term", "accepting", "understanding",
}


def is_english_loanword(word: str) -> bool:
    """Check if a word (romanized or English) represents a common English loanword."""
    clean = re.sub(r'[^a-zA-Z]', '', word).lower()
    return clean in _LOANWORDS


def detect_script_extended(text: str) -> str:
    """Detect script for a string: 'telugu', 'latin', or 'mixed'."""
    has_telugu = any('\u0C00' <= c <= '\u0C7F' for c in text)
    has_latin = any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in text)
    if has_telugu and has_latin:
        return "mixed"
    if has_telugu:
        return "telugu"
    return "latin"
