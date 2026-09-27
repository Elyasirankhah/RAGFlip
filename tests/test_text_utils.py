from utils.text_utils import clean_text, split_into_sentences


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \n\t world") == "hello world"


def test_clean_text_strips_ends():
    assert clean_text("  padded  ") == "padded"


def test_split_into_sentences_keeps_punctuation():
    sentences = split_into_sentences("Hello world. How are you?")
    assert sentences == ["Hello world.", "How are you?"]


def test_split_into_sentences_without_terminator():
    assert split_into_sentences("Just a fragment") == ["Just a fragment"]


def test_split_into_sentences_empty():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []
