"""A small web page for PolySub: python -m polysub.web, then open http://127.0.0.1:7860

Upload a video, choose the languages, and download the subtitles (or the video with them burned in).
The speech model stays loaded between files, so only the first job waits for it to load.
"""
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from . import languages
from .asr import EnginePool
from .pipeline import Options, run
from .translate import api_key_available

POOL = EnginePool()
LANGUAGE_CHOICES = sorted(((name, code) for code, name in languages.NAMES.items()), key=lambda c: c[0])

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#title h1 { margin-bottom: 0; }
#title p { margin-top: 4px; opacity: .8; }
"""


def make_subtitles(file, source, target, bilingual, burn, context, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Upload a video or audio file first.")
    work = Path(tempfile.mkdtemp(prefix="polysub-"))
    upload = work / Path(file).name
    shutil.copy(file, upload)
    options = Options(source=None if source == "auto" else source, target=target, bilingual=bilingual,
                      burn=burn, context=context or "", formats=("srt", "vtt"), output_dir=work)
    try:
        result = run(upload, options, POOL, progress=lambda stage, f=None: progress(f or 0, desc=stage))
    except Exception as error:  # shown as a friendly message on the page
        raise gr.Error(str(error)) from error

    def flat(cue, language):  # one line for the table; Japanese/Chinese lines join without a space
        return cue.text.replace("\n", " " if languages.uses_spaces(language) else "")

    src, tgt = result.source, result.target
    if result.translated is not None and len(result.translated) == len(result.cues):  # Claude keeps the cues paired
        rows = [[_time(t.start), _time(t.end), flat(o, src), flat(t, tgt)] for o, t in zip(result.cues, result.translated)]
    elif result.translated is not None:  # Whisper's English has its own timing: show the original lines
        join = " " if languages.uses_spaces(src) else ""  # that overlap each English line in time
        rows = [[_time(t.start), _time(t.end),
                 join.join(flat(o, src) for o in result.cues if o.start < t.end and o.end > t.start), flat(t, tgt)]
                for t in result.translated]
    else:
        rows = [[_time(o.start), _time(o.end), flat(o, src), ""] for o in result.cues]
    summary = (f"**{languages.label(result.source)} → {languages.label(result.target)}** · {len(result.cues)} subtitles · "
               f"recognised with {result.engine}" + (f" · translated with {result.translator}" if result.translator else "")
               + (f" · API cost about ${result.cost:.2f}" if result.cost is not None else ""))
    if result.warnings:
        summary += "\n\n" + "\n".join(f"⚠ {w}" for w in result.warnings)
    files = [str(f) for key, f in result.files.items() if key != "video"]
    video = str(result.files["video"]) if "video" in result.files else None
    return summary, rows, files, video


def _time(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:05.2f}"


def build_app():
    with gr.Blocks(title="PolySub") as app:
        with gr.Column(elem_id="title"):
            gr.Markdown("# PolySub\nSubtitles from any language to any language: local speech recognition "
                        "(Qwen3-ASR / Whisper large-v3) + Claude translation.")
        if not api_key_available():
            gr.Markdown("ℹ️ No `ANTHROPIC_API_KEY` set: only original-language and English subtitles are available.")
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Video or audio", type="filepath")
                source = gr.Dropdown([("Detect automatically", "auto")] + LANGUAGE_CHOICES, value="auto", label="Spoken language")
                target = gr.Dropdown(LANGUAGE_CHOICES, value="en", label="Subtitle language")
                context = gr.Textbox(label="About the video (optional)", lines=2,
                                     placeholder="e.g. Cooking show; host Aiko; dish names: oyakodon, karaage")
                with gr.Row():
                    bilingual = gr.Checkbox(label="Bilingual subtitles")
                    burn = gr.Checkbox(label="Burn into video")
                go = gr.Button("Make subtitles", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                table = gr.Dataframe(headers=["Start", "End", "Original", "Translation"], wrap=True, label="Subtitles",
                                     column_widths=["11%", "11%", "39%", "39%"])
                files = gr.File(label="Download", file_count="multiple")
                video = gr.Video(label="Video with subtitles")
        go.click(make_subtitles, [file, source, target, bilingual, burn, context], [summary, table, files, video])
    return app


def main():
    build_app().queue().launch(theme=gr.themes.Soft(primary_hue="rose"), css=CSS)


if __name__ == "__main__":
    main()
