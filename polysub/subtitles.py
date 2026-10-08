"""Subtitle file formats: SRT, WebVTT and bilingual (translation above the original)."""
from . import languages
from .cues import Cue


def format_timestamp(seconds, separator=","):
    """Seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (WebVTT)."""
    millis = int(round(max(seconds, 0) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _clean(cues):
    """Drops empty cues, so numbering never has gaps."""
    return [c for c in cues if c.text.strip()]


def to_srt(cues):
    blocks = []
    for number, cue in enumerate(_clean(cues), start=1):
        blocks.append(f"{number}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text.strip()}\n")
    return "\n".join(blocks)


def to_vtt(cues):
    blocks = ["WEBVTT\n"]
    for cue in _clean(cues):
        blocks.append(f"{format_timestamp(cue.start, '.')} --> {format_timestamp(cue.end, '.')}\n{cue.text.strip()}\n")
    return "\n".join(blocks)


def bilingual(original, translated, source="en"):
    """The translation (up to two lines) with the original underneath on one line, same timing,
    so a bilingual subtitle never fills more than three lines."""
    join = " " if languages.uses_spaces(source) else ""
    return [Cue(t.start, t.end, f"{t.text.strip()}\n{join.join(o.text.split())}") for o, t in zip(original, translated)]


WRITERS = {"srt": to_srt, "vtt": to_vtt}
