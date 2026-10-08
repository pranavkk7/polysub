import pytest

from polysub import cli, pipeline
from polysub.pipeline import Options, PipelineError, run

from conftest import FakeEngine, FakePool, FakeTranslator, words

JAPANESE = words("こんにちは。 0.0 0.8 | 今日は 1.0 1.4 | いい天気ですね。 1.4 2.4")
HINDI = words("नमस्ते। 0.0 0.7 | आप 1.5 1.7 | कैसे 1.7 2.0 | हैं? 2.0 2.4")
ENGLISH = words("Hello. 0.0 0.6 | Nice 1.6 1.9 | weather. 1.9 2.5")


def make_pool(detected="ja"):
    spoken = {"ja": JAPANESE, "hi": HINDI}.get(detected, ENGLISH)
    return FakePool({
        "qwen": FakeEngine("Qwen (fake)", spoken, detected),
        "whisper": FakeEngine("Whisper (fake)", spoken, detected),
    })


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"not really a video")
    return path


def test_detected_japanese_goes_to_whisper_and_is_translated(video, silent_audio):
    pool, translator = make_pool("ja"), FakeTranslator()
    result = run(video, Options(target="en", formats=("srt", "vtt"), bilingual=True, context="Travel vlog"),
                 pool, translator=translator)

    assert pool.requested[:2] == ["whisper", "whisper"]  # measured best for Japanese: no model swap needed
    assert result.source == "ja" and result.engine == "Whisper (fake)"
    assert pool.engines["whisper"].calls[0]["context"] == "Travel vlog"
    assert translator.calls == [("ja", "en", "Travel vlog")]
    assert set(result.files) == {"ja.srt", "ja.vtt", "en.srt", "en.vtt", "en+ja.srt", "en+ja.vtt"}
    english = (video.parent / "episode.en.srt").read_text(encoding="utf-8")
    assert "[en] こんにちは。" in english
    both = (video.parent / "episode.en+ja.srt").read_text(encoding="utf-8")
    # translation first, then the original Japanese on one line underneath
    assert both.splitlines()[-1] == "こんにちは。今日はいい天気ですね。"


def test_detected_english_is_recognised_by_qwen(video, silent_audio):
    pool = make_pool("en")
    result = run(video, Options(target="hi"), pool, translator=FakeTranslator())
    assert pool.requested[:2] == ["whisper", "qwen"]  # detect with Whisper, recognise with Qwen
    assert result.engine == "Qwen (fake)" and result.translated[0].text.startswith("[hi]")


def test_hindi_is_recognised_by_whisper(video, silent_audio):
    pool = make_pool("hi")
    result = run(video, Options(target="en"), pool, translator=FakeTranslator())
    assert result.source == "hi" and result.engine == "Whisper (fake)"
    assert "qwen" not in pool.requested


def test_english_to_any_language(video, silent_audio):
    pool = make_pool("en")
    result = run(video, Options(source="en", target="ml"), pool, translator=FakeTranslator())
    assert result.target == "ml" and result.translated[0].text.startswith("[ml]")
    assert pool.requested == ["qwen"]  # language given, so no detection


def test_same_language_only_transcribes(video, silent_audio):
    result = run(video, Options(source="ja", target="ja"), make_pool(), translator=FakeTranslator())
    assert result.translated is None and set(result.files) == {"ja.srt"}


def test_without_an_api_key_only_english_works(video, silent_audio, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(PipelineError, match="ANTHROPIC_API_KEY"):
        run(video, Options(source="ja", target="hi"), make_pool())

    pool = make_pool("ja")
    result = run(video, Options(source="ja", target="en"), pool)  # falls back to Whisper's own translation
    assert result.translator.startswith("Whisper")
    assert pool.engines["whisper"].calls[-1]["task"] == "translate"


def test_forcing_qwen_for_a_language_it_cannot_time_is_refused(video, silent_audio):
    with pytest.raises(PipelineError, match="cannot time Hindi"):
        run(video, Options(source="hi", engine="qwen"), make_pool(), translator=FakeTranslator())


def test_burn_uses_the_bilingual_subtitles(video, silent_audio, monkeypatch):
    burned = {}
    monkeypatch.setattr(pipeline, "burn_subtitles", lambda v, s, o: burned.update(srt=s) or o)
    result = run(video, Options(source="ja", target="en", bilingual=True, burn=True, formats=("vtt",)),
                 make_pool(), translator=FakeTranslator())
    assert burned["srt"].name == "episode.en+ja.srt"
    assert result.files["video"].name == "episode.subtitled.mp4"


# ---- command line ----
def test_cli_happy_path(video, silent_audio, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ffmpeg_available", lambda: True)
    code = cli.main([str(video), "--to", "English", "-o", str(video.parent / "out")],
                    pool=make_pool("ja"), translator=FakeTranslator())
    out = capsys.readouterr().out
    assert code == 0
    assert "Japanese (ja)" in out and "Whisper (fake)" in out and "about $0.05" in out
    assert (video.parent / "out" / "episode.en.srt").exists()


def test_cli_errors_are_short_and_clear(video, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "ffmpeg_available", lambda: True)
    assert cli.main([str(tmp_path / "missing.mp4")], pool=make_pool()) == 1
    assert "file not found" in capsys.readouterr().err
    assert cli.main([str(video), "--to", "klingon!"], pool=make_pool()) == 2
    assert cli.main([str(video), "--format", "docx"], pool=make_pool()) == 2

    monkeypatch.setattr(cli, "ffmpeg_available", lambda: False)
    assert cli.main([str(video)], pool=make_pool()) == 1
    assert "FFmpeg is required" in capsys.readouterr().err


def test_cli_quiet_prints_nothing(video, silent_audio, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ffmpeg_available", lambda: True)
    assert cli.main([str(video), "--from", "ja", "--to", "ja", "-q"], pool=make_pool()) == 0
    assert capsys.readouterr().out == ""
