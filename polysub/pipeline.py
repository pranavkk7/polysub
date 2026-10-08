"""The full job: audio -> speech recognition -> subtitle cues -> translation -> files.

Shared by the command line and the web app. Every heavy part (engines, translator) can be swapped for a
stand-in, which is how the tests run the whole pipeline without a GPU or an API key.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import languages
from .audio import load_audio
from .burn import burn_subtitles
from .cues import build_cues, characters_per_second
from .subtitles import WRITERS, bilingual
from .translate import ClaudeTranslator, api_key_available


class PipelineError(Exception):
    pass


@dataclass
class Options:
    source: str = None  # language code, or None to detect it
    target: str = "en"
    engine: str = "auto"  # auto | qwen | whisper
    translator: str = "auto"  # auto | claude | whisper (English only, offline)
    formats: tuple = ("srt",)
    bilingual: bool = False
    burn: bool = False
    context: str = ""
    output_dir: Path = None


@dataclass
class Result:
    source: str
    target: str
    engine: str
    translator: str = ""
    cues: list = field(default_factory=list)
    translated: list = None
    files: dict = field(default_factory=dict)
    duration: float = 0.0
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    cost: float = None


def no_progress(stage, fraction=None):
    pass


def choose_translator(options, source, have_translator=False):
    """claude when an API key is set (or a translator is passed in); Whisper's own English translation as
    the offline fallback."""
    if source == options.target:
        return None
    wanted = options.translator
    if wanted == "auto":
        wanted = "claude" if (have_translator or api_key_available()) else ("whisper" if options.target == "en" else None)
        if wanted is None:
            raise PipelineError(
                f"Translating to {languages.name(options.target)} needs Claude: set ANTHROPIC_API_KEY. "
                "Without a key, only English subtitles (Whisper translation) are possible.")
    if wanted == "whisper" and options.target != "en":
        raise PipelineError("The Whisper translator can only translate into English.")
    return wanted


def run(path, options, pool, translator=None, progress=no_progress):
    path = Path(path)
    out_dir = Path(options.output_dir) if options.output_dir else path.parent
    started = time.perf_counter()
    timings = {}

    progress("Reading the audio", None)
    audio = load_audio(path)
    duration = len(audio) / 16000

    # 1. Which language is spoken? Whisper detects about 100 languages from the first 30 seconds.
    source = options.source
    if source is None:
        progress("Detecting the language", None)
        source, _ = pool.get("whisper").detect_language(audio)

    engine_name = options.engine if options.engine != "auto" else languages.engine_for(source)
    if engine_name == "qwen" and source not in languages.QWEN_TIMED:
        raise PipelineError(f"Qwen cannot time {languages.name(source)}; use --engine whisper or auto.")
    method = choose_translator(options, source, have_translator=translator is not None)

    # 2. Speech recognition, in the original language.
    progress(f"Recognising {languages.name(source)} speech", 0.0)
    t = time.perf_counter()
    engine = pool.get(engine_name)
    transcript = engine.transcribe(audio, source, context=options.context,
                                   progress=lambda f: progress(f"Recognising {languages.name(source)} speech", f))
    timings["recognition"] = time.perf_counter() - t
    cues = build_cues(transcript.words, source)
    if not cues:
        raise PipelineError("No speech was found in this file.")

    result = Result(source=source, target=options.target, engine=engine.label, cues=cues, duration=duration)

    # 3. Translation, keeping each cue's timing.
    if method == "claude":
        progress(f"Translating to {languages.name(options.target)}", 0.0)
        t = time.perf_counter()
        translator = translator or ClaudeTranslator()
        result.translated = translator.translate(
            cues, source, options.target, context=options.context,
            progress=lambda f: progress(f"Translating to {languages.name(options.target)}", f))
        result.translator = translator.label
        result.cost = translator.cost_estimate()
        timings["translation"] = time.perf_counter() - t
    elif method == "whisper":
        progress("Translating to English with Whisper", 0.0)
        t = time.perf_counter()
        english = pool.get("whisper").transcribe(audio, source, context=options.context, task="translate",
                                                 progress=lambda f: progress("Translating to English with Whisper", f))
        result.translated = build_cues(english.words, "en")
        result.translator = "Whisper large-v3 (offline, English only)"
        timings["translation"] = time.perf_counter() - t
        if options.bilingual:
            result.warnings.append("Bilingual subtitles need Claude translation (Whisper's English has its own timing).")

    # 4. Files.
    progress("Saving the subtitles", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    sets = {source: cues}
    if result.translated is not None:
        sets[options.target] = result.translated
        if options.bilingual and method == "claude":
            sets[f"{options.target}+{source}"] = bilingual(cues, result.translated, source)
    for suffix, cue_list in sets.items():
        for fmt in options.formats:
            file = out_dir / f"{stem}.{suffix}.{fmt}"
            file.write_text(WRITERS[fmt](cue_list), encoding="utf-8")
            result.files[f"{suffix}.{fmt}"] = file

    if options.burn:
        progress("Adding the subtitles to the video", None)
        # Burn the most useful set: bilingual, else the translation, else the original language.
        chosen = next(s for s in (f"{options.target}+{source}", options.target, source) if s in sets)
        srt = result.files.get(f"{chosen}.srt")
        if srt is None:  # SRT was not among the requested formats
            srt = out_dir / f"{stem}.{chosen}.srt"
            srt.write_text(WRITERS["srt"](sets[chosen]), encoding="utf-8")
        result.files["video"] = burn_subtitles(path, srt, out_dir / f"{stem}.subtitled.mp4")

    fast = sum(1 for c in (result.translated or cues) if characters_per_second(c) > 20)
    if fast:
        result.warnings.append(f"{fast} subtitles are fast to read (over 20 characters per second).")
    timings["total"] = time.perf_counter() - started
    result.timings = timings
    return result
