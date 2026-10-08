"""Command line: python -m polysub VIDEO --to en"""
import argparse
import os
import sys
import time
from pathlib import Path

from . import languages
from .audio import AudioError, ffmpeg_available
from .burn import BurnError
from .pipeline import Options, PipelineError, run
from .translate import TranslationError


def format_duration(seconds):
    """42s, 3m 05s, 1h 02m 09s."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_size(size_bytes):
    """812 B, 4.2 KB, 124.3 MB."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class Console:
    """Tidy terminal output: colour only on a real terminal, and NO_COLOR is respected."""

    def __init__(self, stream=None, quiet=False, color=None):
        self.stream = stream or sys.stdout
        self.quiet = quiet
        self.color = (self.stream.isatty() and "NO_COLOR" not in os.environ) if color is None else color
        self._last_stage = None

    def _paint(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _write(self, text):
        if not self.quiet:
            print(text, file=self.stream)

    def title(self, text):
        self._write(self._paint(text, "1;36"))

    def field(self, label, value):
        self._write(f"  {self._paint(f'{label:<11}', '2')} {value}")

    def success(self, text):
        self._write(self._paint(text, "1;32"))

    def warn(self, text):
        self._write(self._paint(f"  ! {text}", "33"))

    def blank(self):
        self._write("")

    def progress(self, stage, fraction=None):
        """Prints each stage once, then its percentage on the same line."""
        if self.quiet:
            return
        if stage != self._last_stage:
            if self._last_stage is not None:
                print(file=self.stream)
            self._last_stage = stage
        percent = f" {int(fraction * 100):3d}%" if fraction is not None else ""
        print(f"\r{self._paint('>', '36')} {stage}...{percent}", end="", file=self.stream, flush=True)

    def end_progress(self):
        if self._last_stage is not None and not self.quiet:
            print(file=self.stream)
        self._last_stage = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="polysub",
        description="Subtitles from any language to any language: local speech recognition + Claude translation.",
        epilog="Examples: polysub episode.mkv --to en | polysub talk.mp4 --to hi --bilingual --burn",
    )
    p.add_argument("input", type=Path, help="video or audio file")
    p.add_argument("--to", dest="target", default="en", help="subtitle language: code or name (default: en)")
    p.add_argument("--from", dest="source", default="auto", help="spoken language (default: detect)")
    p.add_argument("--engine", default="auto", choices=["auto", "qwen", "whisper"],
                   help="speech recognition engine (default: best for the language)")
    p.add_argument("--translator", default="auto", choices=["auto", "claude", "whisper"],
                   help="claude (any language, needs ANTHROPIC_API_KEY) or whisper (English only, offline)")
    p.add_argument("--format", dest="formats", default="srt", help="srt, vtt or srt,vtt (default: srt)")
    p.add_argument("--bilingual", action="store_true", help="also write subtitles with the original under the translation")
    p.add_argument("--burn", action="store_true", help="also save a copy of the video with the subtitles burned in")
    p.add_argument("--context", default="", help="a sentence about the video, names or terms; improves accuracy")
    p.add_argument("-o", "--output-dir", type=Path, help="folder for the output files (default: next to the input)")
    p.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing except errors")
    return p.parse_args(argv)


def main(argv=None, pool=None, translator=None):
    args = parse_args(argv)
    console = Console(quiet=args.quiet)
    try:
        options = Options(
            source=languages.resolve(args.source),
            target=languages.resolve(args.target) or "en",
            engine=args.engine,
            translator=args.translator,
            formats=tuple(f.strip() for f in args.formats.lower().split(",") if f.strip()),
            bilingual=args.bilingual,
            burn=args.burn,
            context=args.context,
            output_dir=args.output_dir,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if any(f not in ("srt", "vtt") for f in options.formats):
        print("error: --format must be srt, vtt or both", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 1
    if not ffmpeg_available():
        print("error: FFmpeg is required but was not found. Install it and make sure it is on your PATH.", file=sys.stderr)
        return 1

    console.title("PolySub")
    console.field("File", f"{args.input.name} ({format_size(args.input.stat().st_size)})")
    console.field("From", languages.label(options.source) if options.source else "detect automatically")
    console.field("To", languages.label(options.target))
    console.blank()

    if pool is None:
        from .asr import EnginePool

        pool = EnginePool(device=args.device)
    started = time.perf_counter()
    try:
        result = run(args.input, options, pool, translator=translator, progress=console.progress)
    except KeyboardInterrupt:
        console.end_progress()
        print("Stopped. No subtitle files were written.", file=sys.stderr)
        return 130
    except (AudioError, PipelineError, TranslationError, BurnError) as error:
        console.end_progress()
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # a clear message beats a stack trace for a command-line tool
        console.end_progress()
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        pool.close()

    console.end_progress()
    console.blank()
    console.success("Done")
    console.field("Language", languages.label(result.source))
    console.field("Recognised", result.engine)
    if result.translator:
        console.field("Translated", result.translator)
    console.field("Subtitles", f"{len(result.cues)} cue{'' if len(result.cues) == 1 else 's'}")
    console.field("Length", format_duration(result.duration))
    console.field("Time taken", format_duration(time.perf_counter() - started))
    if result.cost is not None:
        console.field("API cost", f"about ${result.cost:.2f}")
    for key, file in result.files.items():
        console.field("Saved" if key == list(result.files)[0] else "", str(file))
    for warning in result.warnings:
        console.warn(warning)
    return 0
