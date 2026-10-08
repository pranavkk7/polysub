"""Accuracy check: how many words (or characters) each engine gets wrong on clips with known transcripts.

    python -m polysub.evaluate DATA_DIR --language ja --engines qwen,whisper

DATA_DIR holds audio files with a matching .txt reference transcript (clip01.wav + clip01.txt).
Reports WER (word error rate) for languages written with spaces and CER (character error rate) for
languages like Japanese and Chinese, where "words" are not separated.
"""
import argparse
import time
import unicodedata
from pathlib import Path

from . import languages
from .audio import SAMPLE_RATE, load_audio

AUDIO_TYPES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".mp4", ".mkv", ".webm"}


def normalize(text):
    """Lower case, no punctuation, single spaces: so "Hello, World!" and "hello world" count as equal."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in text)
    return " ".join(text.split())


def edit_distance(reference, hypothesis):
    """Minimum insertions + deletions + substitutions to turn one sequence into the other."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        current = [i]
        for j, hyp in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def units(text, language):
    text = normalize(text)
    return text.split() if languages.uses_spaces(language) else list(text.replace(" ", ""))


def error_rate(references, hypotheses, language):
    """Corpus WER/CER: total edits divided by total reference words (or characters)."""
    edits = total = 0
    for ref, hyp in zip(references, hypotheses):
        ref_units = units(ref, language)
        edits += edit_distance(ref_units, units(hyp, language))
        total += len(ref_units)
    return edits / total if total else 0.0


def find_clips(data_dir):
    clips = []
    for audio in sorted(Path(data_dir).iterdir()):
        reference = audio.with_suffix(".txt")
        if audio.suffix.lower() in AUDIO_TYPES and reference.exists():
            clips.append((audio, reference.read_text(encoding="utf-8").strip()))
    return clips


def evaluate(clips, language, engine_names, pool):
    rows = []
    for name in engine_names:
        engine = pool.get(name)
        hypotheses, audio_seconds = [], 0.0
        started = time.perf_counter()
        for audio_path, _ in clips:
            audio = load_audio(audio_path)
            audio_seconds += len(audio) / SAMPLE_RATE
            words = engine.transcribe(audio, language).words
            hypotheses.append((" " if languages.uses_spaces(language) else "").join(w.text.strip() for w in words))
        elapsed = time.perf_counter() - started
        rows.append({
            "engine": engine.label,
            "metric": "WER" if languages.uses_spaces(language) else "CER",
            "error_rate": error_rate([ref for _, ref in clips], hypotheses, language),
            "speed": audio_seconds / elapsed if elapsed else 0.0,  # seconds of audio per second of work
        })
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="polysub.evaluate", description=__doc__.split("\n\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("--language", required=True, help="language code of the clips, e.g. ja")
    p.add_argument("--engines", default="qwen,whisper", help="comma-separated: qwen, whisper")
    args = p.parse_args(argv)

    from .asr import EnginePool

    language = languages.resolve(args.language)
    clips = find_clips(args.data_dir)
    if not clips:
        print(f"error: no audio files with matching .txt transcripts in {args.data_dir}")
        return 1
    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    if "qwen" in names and language not in languages.QWEN_TIMED:
        names.remove("qwen")
        print(f"Skipping Qwen: it cannot time {languages.name(language)}.")
    pool = EnginePool()
    try:
        rows = evaluate(clips, language, names, pool)
    finally:
        pool.close()
    print(f"\n{languages.label(language)}, {len(clips)} clips\n")
    print(f"| Engine | {rows[0]['metric']} (lower is better) | Speed (x real time) |")
    print("|---|---|---|")
    for row in rows:
        print(f"| {row['engine']} | {row['error_rate'] * 100:.1f}% | {row['speed']:.1f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
