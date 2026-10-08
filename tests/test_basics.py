import numpy as np
import pytest

from polysub import languages
from polysub.audio import chunk_bounds
from polysub.cues import Cue
from polysub.evaluate import edit_distance, error_rate, normalize
from polysub.subtitles import bilingual, format_timestamp, to_srt, to_vtt


# ---- languages ----
def test_resolve_accepts_codes_names_and_auto():
    assert languages.resolve("ja") == languages.resolve("JA") == languages.resolve("Japanese") == "ja"
    assert languages.resolve("auto") is None and languages.resolve(None) is None
    assert languages.resolve("sw") == "sw"  # unknown to our table, but Whisper may know it
    with pytest.raises(ValueError):
        languages.resolve("klingon!")


def test_each_language_goes_to_the_engine_that_can_time_it():
    assert languages.engine_for("en") == "qwen"  # measured: Qwen 3.8% vs Whisper 4.2% WER
    assert languages.engine_for("ja") == "whisper"  # measured: Whisper 3.8% vs Qwen 5.4% CER
    assert languages.engine_for("ko") == "qwen"  # not measured yet: published benchmarks
    assert languages.engine_for("hi") == "whisper"  # Qwen recognises Hindi but cannot time it
    assert languages.engine_for("ml") == "whisper"
    assert languages.from_name("Japanese") == "ja"


# ---- audio chunking ----
def test_short_audio_is_one_chunk():
    assert chunk_bounds(np.ones(16000 * 60, dtype=np.float32)) == [(0, 16000 * 60)]


def test_long_audio_is_cut_in_the_quietest_moment():
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.3, 16000 * 400).astype(np.float32)
    audio[16000 * 135:16000 * 136] = 0  # a one-second silence inside the search window
    bounds = chunk_bounds(audio, max_seconds=150, search_seconds=30)
    assert abs(bounds[0][1] / 16000 - 135.5) < 0.6
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    assert all(b - a <= 150 * 16000 for a, b in bounds)
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))


# ---- subtitle files ----
def test_timestamps_for_srt_and_vtt():
    assert format_timestamp(3725.5) == "01:02:05,500"
    assert format_timestamp(0.0004, ".") == "00:00:00.000"


def test_srt_skips_empty_cues_without_gaps_in_numbering():
    srt = to_srt([Cue(0, 1, "First"), Cue(1, 2, "  "), Cue(2, 3.25, "Second\nline")])
    assert srt == "1\n00:00:00,000 --> 00:00:01,000\nFirst\n\n2\n00:00:02,000 --> 00:00:03,250\nSecond\nline\n"


def test_vtt_has_a_header_and_dot_timestamps():
    vtt = to_vtt([Cue(1, 2, "Hello")])
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:01.000 --> 00:00:02.000\nHello" in vtt


def test_bilingual_puts_the_translation_above_the_original():
    cues = bilingual([Cue(0, 1, "こんにちは")], [Cue(0, 1, "Hello")])
    assert cues[0].text == "Hello\nこんにちは"


# ---- evaluation metrics ----
def test_normalize_ignores_case_and_punctuation():
    assert normalize("Hello, World!") == normalize("hello world")
    assert normalize("「こんにちは」。") == "こんにちは"


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], ["a"]) == 1


def test_wer_for_spaced_languages_and_cer_for_japanese():
    assert error_rate(["the cat sat"], ["the cat sat"], "en") == 0
    assert error_rate(["the cat sat down"], ["the bat sat"], "en") == 0.5  # 1 substitution + 1 deletion / 4
    assert error_rate(["今日は晴れ"], ["今日は雨"], "ja") == pytest.approx(2 / 5)
