"""Subtitle translation with Claude.

Cues are sent in batches with their numbers. Claude returns exactly one translation per number (enforced
with a JSON schema), so the translated subtitles keep the original timing. Each batch also sees the last
few translated cues, so names and tone stay consistent across the whole video.
"""
import json
import os

from . import languages
from .cues import Cue, wrap_text

DEFAULT_MODEL = os.environ.get("POLYSUB_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("POLYSUB_EFFORT", "medium")
BATCH_SIZE = 40
CONTEXT_CUES = 6
# USD per million input / output tokens, for the cost estimate shown after a run.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}

SYSTEM_PROMPT = """You are a professional subtitle translator.

You receive numbered subtitle cues from a video. Translate each cue into the target language so it reads \
naturally as a subtitle:
- Keep the meaning and tone (casual, formal, joking) of the original. Translate idioms naturally, not word \
for word.
- Keep each translation short enough to read while it is on screen; condense filler words if needed.
- Translate every cue on its own line number. Never merge, split, skip or reorder cues: return one \
translation for every id you receive, even if a cue is a single word or sound (translate or keep it as is).
- Keep names of people and places consistent with the earlier translations shown to you.
- If a cue is already in the target language, keep it unchanged.
Return only the JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


class TranslationError(Exception):
    pass


def api_key_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class ClaudeTranslator:
    label = "Claude"

    def __init__(self, client=None, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=300.0, max_retries=3)
        self.client = client
        self.model = model
        self.effort = effort
        self.label = f"Claude ({model})"
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    def translate(self, cues, source, target, context="", progress=None):
        """New cues with the same timing and the text translated from `source` to `target` (codes)."""
        texts = [c.text.replace("\n", " ") for c in cues]
        translated = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = list(enumerate(texts[start:start + BATCH_SIZE], start=start + 1))
            earlier = list(zip(texts[max(0, start - CONTEXT_CUES):start],
                               translated[max(0, start - CONTEXT_CUES):start]))
            translated.extend(self._translate_batch(batch, earlier, source, target, context))
            if progress:
                progress(min((start + BATCH_SIZE) / len(texts), 1.0))
        return [Cue(c.start, c.end, wrap_text(t, target)) for c, t in zip(cues, translated)]

    def _translate_batch(self, batch, earlier, source, target, context):
        prompt = [f"Translate these subtitle cues from {languages.name(source)} to {languages.name(target)}."]
        if context:
            prompt.append(f"About the video (from the user): {context}")
        if earlier:
            prompt.append("The previous cues and how they were translated (for consistency, do not return them):")
            prompt.extend(f"- {src} => {dst}" for src, dst in earlier)
        prompt.append("Cues to translate:")
        prompt.append(json.dumps([{"id": i, "text": t} for i, t in batch], ensure_ascii=False))
        message = "\n".join(prompt)

        wanted = [i for i, _ in batch]
        for attempt in range(2):
            result = self._request(message if attempt == 0 else message + (
                f"\n\nReturn exactly {len(wanted)} translations, one for each id: {wanted}."))
            by_id = {item["id"]: item["text"].strip() for item in result.get("translations", [])}
            if all(i in by_id for i in wanted):
                return [by_id[i] for i in wanted]
        missing = [i for i in wanted if i not in by_id]
        raise TranslationError(f"Claude did not return translations for cues {missing[:10]}.")

    def _request(self, message):
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            # If the model declines on safety grounds, the API retries on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage["requests"] += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["input_tokens"] += (getattr(usage, "input_tokens", 0) or 0) + (
                getattr(usage, "cache_read_input_tokens", 0) or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        if response.stop_reason == "refusal":
            raise TranslationError("Claude declined to translate part of this video.")
        if response.stop_reason == "max_tokens":
            raise TranslationError("A translation batch was too long. Try again.")
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise TranslationError("Claude returned a reply that was not valid JSON.") from error

    def cost_estimate(self):
        """Approximate USD cost of this run, or None for a model without a known price."""
        prices = PRICES.get(self.model)
        if not prices:
            return None
        return (self.usage["input_tokens"] * prices[0] + self.usage["output_tokens"] * prices[1]) / 1_000_000
