"""Speech recognition engines. Both return timed words in the original language.

- QwenEngine: Qwen3-ASR-1.7B for the text + Qwen3-ForcedAligner-0.6B for word timings (11 languages).
- WhisperEngine: Whisper large-v3 through faster-whisper, with its own word timings (about 100 languages).

The heavy libraries are imported only when an engine is created, so the rest of the tool (and its tests)
works without a GPU or the models.
"""
from dataclasses import dataclass, field

from . import languages
from .audio import SAMPLE_RATE, chunk_bounds
from .cues import Word, attach_punctuation, japanese_tokenizer, merge_pieces, resegment

PUNCTUATION_PROMPTS = {
    "ja": "こんにちは。今日は、いい天気ですね。それでは、始めましょう。",
    "zh": "你好。今天天气很好，我们开始吧。",
    "yue": "你好。今日天氣好好，我哋開始啦。",
}

QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
WHISPER_MODEL = "large-v3"
QWEN_CHUNK_SECONDS = 170  # the aligner handles at most 180 s per call
MIN_MISSED_SECONDS = 1.0  # shorter stretches of "speech" without words are breaths and noises


@dataclass
class Transcript:
    language: str
    words: list = field(default_factory=list)
    engine: str = ""


def missed_spans(speech, words, min_seconds=MIN_MISSED_SECONDS):
    """Stretches the voice detector marked as speech but that produced no words at all.

    Whisper sometimes skips a whole sentence in the middle of its 30-second window (seen with the
    Japanese punctuation prompt on a sentence of unusual names). Those stretches are recognised again
    on their own. `speech` is a list of (start, end) seconds; a stretch counts as covered when any
    word's midpoint falls inside it.
    """
    mids = [(w.start + w.end) / 2 for w in words]
    return [(a, b) for a, b in speech if b - a >= min_seconds and not any(a <= m <= b for m in mids)]


def _free_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class QwenEngine:
    label = "Qwen3-ASR-1.7B + ForcedAligner-0.6B"

    def __init__(self, device="cuda:0"):
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            QWEN_ASR_MODEL,
            dtype=dtype,
            device_map=device,
            max_new_tokens=4096,
            forced_aligner=QWEN_ALIGNER_MODEL,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device),
        )

    def transcribe(self, audio, language, context="", progress=None):
        bounds = chunk_bounds(audio, max_seconds=QWEN_CHUNK_SECONDS)
        words, detected = [], None
        for index, (start, end) in enumerate(bounds):
            result = self.model.transcribe(
                audio=(audio[start:end], SAMPLE_RATE),
                context=context,
                language=languages.name(language) if language else None,
                return_time_stamps=True,
            )[0]
            offset = start / SAMPLE_RATE
            items = getattr(result.time_stamps, "items", None) or []
            tokens = [Word(item.text, item.start_time + offset, item.end_time + offset) for item in items]
            words.extend(attach_punctuation(result.text, tokens))
            # Qwen names the language ("Japanese", or "Japanese,English" for mixed speech).
            detected = detected or languages.from_name((result.language or "").split(",")[0])
            if progress:
                progress((index + 1) / len(bounds))
        return Transcript(language=language or detected or "en", words=words, engine=self.label)

    def close(self):
        self.model = None
        _free_gpu()


class WhisperEngine:
    label = "Whisper large-v3 (faster-whisper)"

    def __init__(self, device="cuda"):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device.startswith("cuda") else "int8"
        self.model = WhisperModel(WHISPER_MODEL, device=device.split(":")[0], compute_type=compute_type)

    def detect_language(self, audio):
        """(code, probability) from the first 30 seconds."""
        code, probability, _ = self.model.detect_language(audio[: 30 * SAMPLE_RATE])
        return code, probability

    def transcribe(self, audio, language, context="", progress=None, task="transcribe"):
        # For Japanese/Chinese, a punctuated example sentence makes Whisper write 。and 、(measured on
        # FLEURS: 94 marks instead of 42, CER 4.7% -> 4.6%). Carrying context between 30 s windows keeps
        # that style for the whole video; the silence threshold stops it inventing text in long pauses.
        style = PUNCTUATION_PROMPTS.get(language) if task == "transcribe" else None
        prompt = " ".join(p for p in (context, style) if p) or None
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,  # skip silence, which is where Whisper tends to invent text
            condition_on_previous_text=bool(style),  # otherwise off: one mistake can't repeat through the file
            hallucination_silence_threshold=2.0 if style else None,
            initial_prompt=prompt,
        )
        pieces = []
        for segment in segments:  # a generator: recognition happens while we iterate
            pieces.extend(Word(w.word, w.start, w.end) for w in (segment.words or []))
            if progress and info.duration:
                progress(min(segment.end / info.duration, 1.0))
        spoken = language or info.language
        pieces = self._recover_missed(audio, pieces, spoken, task, prompt)
        written = spoken if task == "transcribe" else "en"
        if languages.uses_spaces(written):
            words = merge_pieces(pieces)
        else:
            tokenizer = japanese_tokenizer() if written == "ja" else None
            words = resegment(pieces, tokenizer) if tokenizer else pieces
        return Transcript(language=spoken, words=words, engine=self.label)

    def _recover_missed(self, audio, pieces, language, task, prompt):
        """Recognises again, one at a time, any detected speech that produced no words."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        speech = [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in get_speech_timestamps(audio, VadOptions())]
        for start, end in missed_spans(speech, pieces):
            clip = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
            segments, _ = self.model.transcribe(
                clip, language=language, task=task, beam_size=5, word_timestamps=True,
                vad_filter=False,  # the clip is speech already
                condition_on_previous_text=False, initial_prompt=prompt,
            )
            pieces.extend(Word(w.word, w.start + start, w.end + start) for seg in segments for w in (seg.words or []))
        return sorted(pieces, key=lambda w: w.start)

    def close(self):
        self.model = None
        _free_gpu()


ENGINES = {"qwen": QwenEngine, "whisper": WhisperEngine}


class EnginePool:
    """Keeps at most one model on the GPU: 8 GB cannot hold Whisper and both Qwen models at once.
    The web app keeps one pool, so a model loads once and is reused between files."""

    def __init__(self, device="cuda", factories=None):
        self.device = device
        self.factories = factories or ENGINES
        self.loaded = {}

    def get(self, name):
        if name not in self.loaded:
            for other in list(self.loaded):
                self.loaded.pop(other).close()
            # Qwen (transformers) wants "cuda:0"; faster-whisper wants "cuda".
            device = "cuda:0" if name == "qwen" and self.device == "cuda" else self.device
            self.loaded[name] = self.factories[name](device=device)
        return self.loaded[name]

    def close(self):
        for engine in self.loaded.values():
            engine.close()
        self.loaded.clear()
