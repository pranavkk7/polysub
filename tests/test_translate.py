import json

import pytest

from polysub.cues import Cue
from polysub.translate import BATCH_SIZE, ClaudeTranslator, TranslationError

from conftest import FakeClaude, echo_translations


def cues(n):
    return [Cue(i, i + 0.9, f"line {i + 1}") for i in range(n)]


def test_every_cue_is_translated_and_keeps_its_timing():
    client = FakeClaude([echo_translations])
    result = ClaudeTranslator(client=client).translate(cues(3), "ja", "en")
    assert [c.text for c in result] == ["EN: line 1", "EN: line 2", "EN: line 3"]
    assert [(c.start, c.end) for c in result] == [(0, 0.9), (1, 1.9), (2, 2.9)]


def test_long_videos_are_sent_in_batches_with_earlier_cues_as_context():
    client = FakeClaude([echo_translations] * 3)
    result = ClaudeTranslator(client=client).translate(cues(BATCH_SIZE * 2 + 5), "ja", "en", context="A cooking show")
    assert len(client.requests) == 3 and len(result) == BATCH_SIZE * 2 + 5
    second = client.requests[1]["messages"][0]["content"]
    assert f"line {BATCH_SIZE} => EN: line {BATCH_SIZE}" in second  # consistency context from batch 1
    assert "A cooking show" in second
    assert "Japanese to English" in second


def test_request_uses_a_json_schema_and_the_safety_fallback():
    client = FakeClaude([echo_translations])
    ClaudeTranslator(client=client, model="claude-opus-5-5", effort="medium").translate(cues(1), "hi", "en")
    params = client.requests[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"]["effort"] == "medium"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["fallbacks"] == "default"


def test_a_missing_cue_is_asked_for_again_then_reported():
    incomplete = {"translations": [{"id": 1, "text": "one"}]}
    client = FakeClaude([incomplete, echo_translations])
    result = ClaudeTranslator(client=client).translate(cues(2), "ja", "en")
    assert [c.text for c in result] == ["EN: line 1", "EN: line 2"]
    assert "exactly 2 translations" in client.requests[1]["messages"][0]["content"]

    with pytest.raises(TranslationError, match="cues \\[2\\]"):
        ClaudeTranslator(client=FakeClaude([incomplete, incomplete])).translate(cues(2), "ja", "en")


def test_refusals_and_bad_json_become_clear_errors():
    with pytest.raises(TranslationError, match="declined"):
        ClaudeTranslator(client=FakeClaude([({"translations": []}, "refusal")])).translate(cues(1), "ja", "en")
    with pytest.raises(TranslationError, match="not valid JSON"):
        ClaudeTranslator(client=FakeClaude(["not json"])).translate(cues(1), "ja", "en")


def test_translations_are_wrapped_for_the_target_language():
    long_text = "This translated subtitle is definitely longer than forty-two characters"
    client = FakeClaude([{"translations": [{"id": 1, "text": long_text}]}])
    result = ClaudeTranslator(client=client).translate(cues(1), "ja", "en")
    assert "\n" in result[0].text and all(len(line) <= 42 for line in result[0].text.split("\n"))


def test_cost_estimate_from_token_usage():
    translator = ClaudeTranslator(client=FakeClaude([echo_translations]), model="claude-opus-5-5")
    translator.translate(cues(1), "ja", "en")
    assert translator.cost_estimate() == pytest.approx((1000 * 4 + 500 * 20) / 1_000_000)
    assert ClaudeTranslator(client=FakeClaude([]), model="some-other-model").cost_estimate() is None
