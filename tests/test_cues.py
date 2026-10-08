from polysub.cues import Cue, Word, attach_punctuation, build_cues, characters_per_second, fix_timing, wrap_text

from conftest import words


def test_attach_punctuation_moves_marks_onto_the_previous_word():
    tokens = [Word("こんにちは", 0.0, 0.8), Word("元気", 1.0, 1.3), Word("です", 1.3, 1.5), Word("か", 1.5, 1.6)]
    result = attach_punctuation("こんにちは。元気ですか？", tokens)
    assert [w.text for w in result] == ["こんにちは。", "元気", "です", "か？"]

    english = attach_punctuation("Hi, Tom. Ready?", [Word("Hi", 0, 0.2), Word("Tom", 0.3, 0.5), Word("Ready", 0.8, 1.1)])
    assert [w.text for w in english] == ["Hi,", "Tom.", "Ready?"]


def test_attach_punctuation_uses_the_spelling_from_the_transcript():
    result = attach_punctuation("hello there", [Word("hello", 0, 0.3), Word("THERE", 0.4, 0.7)])
    assert [w.text for w in result] == ["hello", "there"]


def test_hyphens_and_apostrophes_inside_words_survive_alignment():
    # The aligner returns "glassfronted" for "glass-fronted"; the subtitle must still say glass-fronted.
    text = "The glass-fronted buildings, men's Super-G."
    tokens = [Word(t, i, i + 0.5) for i, t in enumerate(["The", "glassfronted", "buildings", "mens", "SuperG"])]
    assert [w.text for w in attach_punctuation(text, tokens)] == ["The", "glass-fronted", "buildings,", "men's", "Super-G."]


def test_opening_quotes_go_with_the_next_word_and_unknown_tokens_dont_skip_ahead():
    tokens = [Word("彼は", 0, 0.3), Word("言った", 0.3, 0.6), Word("はい", 0.7, 0.9)]
    assert [w.text for w in attach_punctuation("彼は言った。「はい」", tokens)] == ["彼は", "言った。", "「はい」"]

    tokens = [Word("one", 0, 0.2), Word("zzz", 0.3, 0.4), Word("two", 0.5, 0.7)]
    # "zzz" isn't in the transcript: it is kept, and the words after it are still matched correctly.
    assert [w.text for w in attach_punctuation("one, two.", tokens)] == ["one", "zzz,", "two."]


def test_whisper_sub_word_pieces_are_merged_into_words():
    from polysub.cues import merge_pieces

    # English translated from Japanese arrives as pieces; a piece without a leading space continues a word.
    pieces = [Word(" the", 0, 0.2), Word(" H", 0.3, 0.4), Word("ons", 0.4, 0.5), Word("hu", 0.5, 0.6),
              Word(",", 0.6, 0.6), Word(" 7", 0.7, 0.8), Word(",", 0.8, 0.8), Word("000", 0.8, 1.0),
              Word("Next", 2.0, 2.3)]  # no space, but long after: a new word, not part of "000"
    assert [w.text for w in merge_pieces(pieces)] == ["the", "Honshu,", "7,000", "Next"]
    assert merge_pieces(pieces)[1].start == 0.3 and merge_pieces(pieces)[1].end == 0.6


def test_japanese_character_pieces_are_regrouped_into_words():
    from polysub.cues import resegment

    pieces = [Word("尊", 1.0, 1.2), Word("厳", 1.2, 1.4), Word("と", 1.4, 1.5), Word("礼", 1.5, 1.7),
              Word("節", 1.7, 1.9), Word("。", 1.9, 1.9)]
    words = resegment(pieces, lambda text: ["尊厳", "と", "礼節", "。"])
    assert [(w.text, w.start, w.end) for w in words] == [("尊厳", 1.0, 1.4), ("と", 1.4, 1.5), ("礼節。", 1.5, 1.9)]


def test_lines_break_between_words_never_inside_one():
    from polysub.cues import wrap_words

    words = [Word(t, 0, 1) for t in ["場に", "ふさわしい", "敬意を", "払い、", "尊厳と", "礼節を"]]
    first, second = wrap_words(words, "ja").split("\n")
    assert first == "場にふさわしい敬意を払い、"  # breaks after the comma, between two words
    assert second == "尊厳と礼節を"


def test_whisper_hyphen_pieces_are_joined_without_a_space():
    from polysub.cues import join_words

    pieces = [Word(" glass", 0, 0.2), Word("-fronted", 0.2, 0.5), Word(" buildings", 0.5, 0.9)]
    assert join_words(pieces, "en") == "glass-fronted buildings"


def test_a_pause_starts_a_new_cue():
    cues = build_cues(words("Good 0.0 0.3 | morning 0.35 0.8 | How 2.0 2.2 | are 2.25 2.4 | you 2.45 2.7"), "en")
    assert [c.text for c in cues] == ["Good morning", "How are you"]


def test_a_sentence_end_starts_a_new_cue_once_the_cue_has_some_length():
    spec = "This is the first sentence. 0.0 1.5 | And this is the second one. 1.6 3.0"
    assert len(build_cues(words(spec), "en")) == 2
    assert len(build_cues(words("Yes. 0.0 0.3 | I agree 0.4 0.9"), "en")) == 1  # too short to stand alone


def test_long_speech_is_split_and_wrapped_into_two_lines():
    long_words = [Word(f"word{i:02d}", i * 0.3, i * 0.3 + 0.25) for i in range(40)]
    cues = build_cues(long_words, "en")
    assert len(cues) > 1
    for cue in cues:
        lines = cue.text.split("\n")
        assert len(lines) <= 2
        assert all(len(line) <= 42 for line in lines)
        assert cue.duration <= 7.0 + 1e-9


def test_wrap_text_balances_lines_and_prefers_a_comma():
    assert wrap_text("short line", "en") == "short line"
    wrapped = wrap_text("I went to the market early, and then I came straight back home", "en")
    assert wrapped.split("\n")[0].endswith(",")
    japanese = wrap_text("今日はとても良い天気ですね、散歩に行きましょう", "ja")
    first, second = japanese.split("\n")
    assert first.endswith("、")
    assert len(first) <= 16 and len(second) <= 16


def test_short_cues_stay_on_screen_longer_but_never_overlap():
    cues = fix_timing([Cue(0.0, 0.3, "Hi"), Cue(0.6, 0.9, "Hello"), Cue(5.0, 5.2, "Bye")])
    assert cues[0].end <= cues[1].start - 0.04 + 1e-9
    assert cues[2].end == 6.0  # nothing after it, so it gets the full second
    assert all(c.end > c.start for c in cues)


def test_characters_per_second():
    assert characters_per_second(Cue(0, 2, "ten chars!")) == 5
