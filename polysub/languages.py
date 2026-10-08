"""Languages: names, codes, and which speech engine handles each one.

Engine choice, in order of evidence:
1. Our own measurements (polysub.evaluate on FLEURS dev, 90 clips, October 2026) where we have them:
   English: Qwen 3.8% WER vs Whisper 4.2%  -> Qwen
   Japanese: Whisper 3.8% CER vs Qwen 5.4% -> Whisper (Qwen made real mistakes, not spelling variants)
2. Otherwise the published benchmarks: Qwen3-ASR-1.7B beats Whisper large-v3 on multilingual averages
   (FLEURS 4.90 vs 5.27), so the other languages its forced aligner can time go to Qwen.
3. Everything else (Hindi, Malayalam, Tamil, Arabic, ...) goes to Whisper large-v3, which covers about
   100 languages and gives its own word timings.
"""

NAMES = {
    "en": "English", "ja": "Japanese", "zh": "Chinese", "yue": "Cantonese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "hi": "Hindi", "ml": "Malayalam", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "mr": "Marathi",
    "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "ne": "Nepali", "si": "Sinhala",
    "ar": "Arabic", "fa": "Persian", "he": "Hebrew", "tr": "Turkish", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian", "ms": "Malay", "tl": "Filipino", "nl": "Dutch", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek", "uk": "Ukrainian",
}

# The languages Qwen3-ForcedAligner-0.6B can time (the ASR model itself knows 30).
QWEN_TIMED = {"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}

# Written without spaces between words.
NO_SPACES = {"ja", "zh", "yue", "th", "lo", "my", "km"}

# Characters per subtitle line, from common subtitling style guides (two lines at most).
LINE_CHARS = {"ja": 13, "zh": 16, "yue": 16, "ko": 16, "th": 35}
DEFAULT_LINE_CHARS = 42

_BY_NAME = {name.lower(): code for code, name in NAMES.items()}


def resolve(language):
    """A language code from a code or an English name: 'ja', 'JA' and 'Japanese' all give 'ja'.
    None, '' and 'auto' mean "detect it"."""
    if language is None or str(language).strip().lower() in ("", "auto"):
        return None
    value = str(language).strip().lower()
    if value in NAMES:
        return value
    if value in _BY_NAME:
        return _BY_NAME[value]
    if value.isalpha() and 2 <= len(value) <= 3:
        return value  # a code we have no name for; Whisper may still know it
    raise ValueError(f"Unknown language: {language!r}. Use a code like 'ja' or a name like 'Japanese'.")


def name(code):
    return NAMES.get(code, code or "unknown")


def label(code):
    """'ja' -> 'Japanese (ja)'."""
    return f"{NAMES[code]} ({code})" if code in NAMES else (code or "unknown")


# Measured winners (see the module docstring and the README); re-run polysub.evaluate to add languages.
MEASURED_BEST = {"en": "qwen", "ja": "whisper"}


def engine_for(code):
    if code in MEASURED_BEST:
        return MEASURED_BEST[code]
    return "qwen" if code in QWEN_TIMED else "whisper"


def line_chars(code):
    return LINE_CHARS.get(code, DEFAULT_LINE_CHARS)


def uses_spaces(code):
    return code not in NO_SPACES


def from_name(language_name):
    """Qwen reports languages by name ('Japanese'); turn that back into a code."""
    if not language_name:
        return None
    return _BY_NAME.get(str(language_name).strip().lower())
