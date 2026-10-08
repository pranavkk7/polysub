import json
from types import SimpleNamespace

import numpy as np
import pytest

from polysub.asr import Transcript
from polysub.cues import Word


def words(spec):
    """'Hello 0.0 0.4 | world. 0.5 0.9' -> [Word, Word]"""
    out = []
    for part in spec.split("|"):
        text, start, end = part.strip().rsplit(" ", 2)
        out.append(Word(text, float(start), float(end)))
    return out


class FakeEngine:
    def __init__(self, label, transcript_words, detected="ja"):
        self.label = label
        self.transcript_words = transcript_words
        self.detected = detected
        self.calls = []
        self.closed = False

    def detect_language(self, audio):
        return self.detected, 0.99

    def transcribe(self, audio, language, context="", progress=None, task="transcribe"):
        self.calls.append({"language": language, "context": context, "task": task})
        if progress:
            progress(1.0)
        return Transcript(language=language or self.detected, words=list(self.transcript_words), engine=self.label)

    def close(self):
        self.closed = True


class FakePool:
    def __init__(self, engines):
        self.engines = engines
        self.requested = []

    def get(self, name):
        self.requested.append(name)
        return self.engines[name]

    def close(self):
        pass


class FakeTranslator:
    label = "Fake translator"

    def __init__(self):
        self.calls = []

    def translate(self, cues, source, target, context="", progress=None):
        from polysub.cues import Cue

        self.calls.append((source, target, context))
        return [Cue(c.start, c.end, f"[{target}] {c.text}") for c in cues]

    def cost_estimate(self):
        return 0.05


class FakeClaude:
    """Stands in for anthropic.Anthropic(): returns scripted JSON replies and records each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.requests.append(params)
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(params)
        stop_reason = "end_turn"
        if isinstance(reply, tuple):
            reply, stop_reason = reply
        text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


def echo_translations(params):
    """A reply that translates every cue in the request by prefixing 'EN:'."""
    cues = json.loads(params["messages"][0]["content"].split("Cues to translate:\n", 1)[1].split("\n\n")[0])
    return {"translations": [{"id": c["id"], "text": f"EN: {c['text']}"} for c in cues]}


@pytest.fixture
def silent_audio(monkeypatch):
    """Skips FFmpeg: the pipeline 'loads' 30 seconds of silence."""
    import polysub.pipeline

    monkeypatch.setattr(polysub.pipeline, "load_audio", lambda path: np.zeros(16000 * 30, dtype=np.float32))
