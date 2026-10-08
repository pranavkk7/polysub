"""Timed words -> subtitle cues, following common subtitling rules.

A new cue starts when the current one would get too long (two lines), last too long (7 s), when the
speaker pauses, or after a sentence ends. Text is wrapped into at most two balanced lines, very short
cues are kept on screen a little longer, and cues never overlap.
"""
from dataclasses import dataclass

from . import languages

SENTENCE_END = (".", "!", "?", "。", "！", "？", "…", "।")
CLAUSE_END = (",", ";", ":", "、", "，", "；", "：")
PUNCTUATION_BONUS = 4  # a line break right after punctuation reads better, even if lines are a little uneven


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self):
        return self.end - self.start


OPENING = "「『（(“‘\"'[〈《"
GLUE_TO_PREVIOUS = "-'’"  # Whisper sometimes splits "glass-fronted" into "glass" + "-fronted"


def join_words(words, language):
    parts = [w.text.strip() for w in words if w.text.strip()]
    if not languages.uses_spaces(language):
        return "".join(parts)
    text = ""
    for part in parts:
        glue = text and (part[0] in GLUE_TO_PREVIOUS or not any(ch.isalnum() for ch in part))
        text += part if (glue or not text) else " " + part
    return text


def merge_pieces(pieces, max_gap=0.3):
    """Whisper splits text by the *spoken* language, so English translated from Japanese arrives in
    sub-word pieces (" H", "ons", "hu"). A piece without a leading space continues the previous word."""
    words = []
    for piece in pieces:
        if not piece.text.strip():
            continue
        continues = (words and not piece.text[0].isspace() and piece.start - words[-1].end < max_gap) or (
            words and not any(ch.isalnum() for ch in piece.text))
        if continues:
            words[-1] = Word(words[-1].text + piece.text.strip(), words[-1].start, piece.end)
        else:
            words.append(Word(piece.text.strip(), piece.start, piece.end))
    return words


def resegment(pieces, tokenize):
    """Regroups character-level pieces (Whisper's Japanese) into real words using `tokenize` (e.g. the
    nagisa word splitter), giving each word the time of its first and last character, so subtitles never
    break inside a word. Punctuation stays with the word before it."""
    chars, times = [], []
    for piece in pieces:
        text = piece.text.strip()
        for i, ch in enumerate(text):
            share = (piece.end - piece.start) / len(text)
            chars.append(ch)
            times.append((piece.start + share * i, piece.start + share * (i + 1)))
    text = "".join(chars)
    words, cursor = [], 0
    for token in tokenize(text):
        position = text.find(token, cursor) if token.strip() else -1
        if position < 0:
            continue
        if position > cursor and words:  # anything the tokenizer skipped stays with the previous word
            words[-1] = Word(words[-1].text + text[cursor:position], words[-1].start, times[position - 1][1])
        end = position + len(token)
        if words and not any(ch.isalnum() for ch in token):
            words[-1] = Word(words[-1].text + token, words[-1].start, times[end - 1][1])
        else:
            words.append(Word(token, times[position][0], times[end - 1][1]))
        cursor = end
    if words and cursor < len(text):
        words[-1] = Word(words[-1].text + text[cursor:], words[-1].start, times[-1][1])
    return words


def japanese_tokenizer():
    """nagisa's word splitter (installed with qwen-asr), or None if it isn't available."""
    try:
        import nagisa
    except ImportError:
        return None
    return lambda text: nagisa.tagging(text).words


def _match(full_text, start, token):
    """Where `token` appears in `full_text` from `start`, ignoring case and allowing punctuation inside
    it (the aligner turns "glass-fronted" into "glassfronted"). Returns the end index, or None."""
    wanted = [ch.lower() for ch in token if ch.isalnum()]
    i, k = start, 0
    while k < len(wanted) and i < len(full_text):
        ch = full_text[i]
        if ch.lower() == wanted[k]:
            k += 1
        elif ch.isalnum() or ch.isspace() or k == 0:
            return None
        i += 1
    return i if k == len(wanted) else None


def attach_punctuation(full_text, tokens, window=40):
    """Forced aligners time the words but drop punctuation (and hyphens inside words). This rebuilds each
    word as it is written in the full transcript, with the punctuation around it, so subtitles read
    correctly and can break at sentence ends. Opening quotes and brackets go with the next word."""
    words, cursor = [], 0
    for token in tokens:
        if not any(ch.isalnum() for ch in token.text):
            continue
        found = None
        for start in range(cursor, min(cursor + window, len(full_text))):  # a token the text doesn't have
            end = _match(full_text, start, token.text)                     # must not make us skip ahead
            if end is not None:
                found = (start, end)
                break
        if found is None:
            words.append(Word(token.text.strip(), token.start, token.end))
            continue
        start, end = found
        between = [ch for ch in full_text[cursor:start] if not ch.isspace() and not ch.isalnum()]
        prefix = "".join(ch for ch in between if ch in OPENING)
        if words:
            words[-1].text += "".join(ch for ch in between if ch not in OPENING)
        words.append(Word(prefix + full_text[start:end], token.start, token.end))
        cursor = end
    tail = "".join(ch for ch in full_text[cursor:] if not ch.isspace() and not ch.isalnum())
    if words and tail:
        words[-1].text += tail
    return words


def wrap_text(text, language, max_chars=None):
    """Splits text into at most two lines of similar length, preferring a break after a comma."""
    max_chars = max_chars or languages.line_chars(language)
    text = " ".join(text.split()) if languages.uses_spaces(language) else text.strip()
    if len(text) <= max_chars:
        return text
    if languages.uses_spaces(language):
        spaces = [i for i, ch in enumerate(text) if ch == " "]
        if not spaces:
            return text
        best = min(spaces, key=lambda i: (max(i, len(text) - i - 1) - (PUNCTUATION_BONUS if text[i - 1] in CLAUSE_END else 0)))
        return text[:best] + "\n" + text[best + 1:]
    middle = len(text) // 2
    candidates = [i + 1 for i, ch in enumerate(text[:-1]) if ch in CLAUSE_END + SENTENCE_END]
    best = min(candidates, key=lambda i: abs(i - middle)) if candidates else middle
    if abs(best - middle) > len(text) // 4:
        best = middle
    return text[:best] + "\n" + text[best:]


def wrap_words(words, language, max_chars=None):
    """Like wrap_text, but only ever breaks between words, so a line never ends inside a word."""
    max_chars = max_chars or languages.line_chars(language)
    full = join_words(words, language)
    if len(full) <= max_chars:
        return full
    if len(words) < 2:
        return wrap_text(full, language, max_chars)
    best = None
    for k in range(1, len(words)):
        first, second = join_words(words[:k], language), join_words(words[k:], language)
        score = max(len(first), len(second)) - (PUNCTUATION_BONUS if first.endswith(CLAUSE_END + SENTENCE_END) else 0)
        if best is None or score < best[0]:
            best = (score, first, second)
    return f"{best[1]}\n{best[2]}"


def build_cues(words, language, max_lines=2, max_duration=7.0, pause=0.8, min_duration=1.0):
    limit = languages.line_chars(language) * max_lines
    cues, current = [], []

    def flush():
        cues.append(Cue(current[0].start, current[-1].end, wrap_words(current, language)))

    for word in words:
        if not word.text.strip():
            continue
        if current:
            longer = join_words(current + [word], language)
            so_far = join_words(current, language)
            if (len(longer) > limit
                    or word.end - current[0].start > max_duration
                    or word.start - current[-1].end >= pause
                    or (current[-1].text.strip().endswith(SENTENCE_END) and len(so_far) >= limit * 0.25)):
                flush()
                current = []
        current.append(word)
    if current:
        flush()
    fix_timing(cues, min_duration)
    return cues


def fix_timing(cues, min_duration=1.0, gap=0.04):
    """Very short cues stay on screen for at least `min_duration` when there is room, and a cue always
    ends before the next one starts."""
    for i, cue in enumerate(cues):
        next_start = cues[i + 1].start if i + 1 < len(cues) else None
        if cue.duration < min_duration:
            wanted = cue.start + min_duration
            cue.end = max(cue.end, min(wanted, next_start - gap) if next_start is not None else wanted)
        if next_start is not None and cue.end > next_start - gap:
            cue.end = max(cue.start + 0.1, next_start - gap)
    return cues


def characters_per_second(cue):
    return len(cue.text.replace("\n", "")) / max(cue.duration, 0.001)
