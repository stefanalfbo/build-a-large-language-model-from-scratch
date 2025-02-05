import pytest

from src.simple_tokenizer import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    basic_tokenizer,
    create_vocabulary,
    the_verdict_vocabulary,
)


def test_basic_tokenizer():
    # Arrange
    text = "Hello, world. Is this-- a test?"
    expected = ["Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"]

    # Act
    preprocessed = basic_tokenizer(text)

    # Assert
    assert preprocessed == expected


def test_create_vocabulary():
    # Arrange
    preprocessed = ["Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"]
    expected = {
        ",": 0,
        "--": 1,
        ".": 2,
        "?": 3,
        "Hello": 4,
        "Is": 5,
        "a": 6,
        "test": 7,
        "this": 8,
        "world": 9,
        "<|endoftext|>": 10,
        "<|unknown|>": 11,
    }

    # Act
    vocabulary = create_vocabulary(preprocessed)

    # Assert
    assert vocabulary == expected


def test_SimpleTokenizerV1_encode():
    # Arrange
    expected = [
        1,
        56,
        2,
        850,
        988,
        602,
        533,
        746,
        5,
        1126,
        596,
        5,
        1,
        67,
        7,
        38,
        851,
        1108,
        754,
        793,
        7,
    ]
    vocabulary = the_verdict_vocabulary()
    tokenizer = SimpleTokenizerV1(vocabulary)
    text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

    # Act
    ids = tokenizer.encode(text)

    # Assert
    assert ids == expected


def test_SimpleTokenizerV1_decode():
    # Arrange
    expected = """" It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    vocabulary = the_verdict_vocabulary()
    tokenizer = SimpleTokenizerV1(vocabulary)
    ids = [
        1,
        56,
        2,
        850,
        988,
        602,
        533,
        746,
        5,
        1126,
        596,
        5,
        1,
        67,
        7,
        38,
        851,
        1108,
        754,
        793,
        7,
    ]

    # Act
    text = tokenizer.decode(ids)

    # Assert
    assert text == expected


def test_SimpleTokenizerV1_key_error():
    """The problem is that the word "Hello" was not used in the "The Verdict" short
    story. Hence, it is not in the vocabulary."""

    # Arrange
    vocabulary = the_verdict_vocabulary()
    tokenizer = SimpleTokenizerV1(vocabulary)
    text = "Hello, do you like tea?"

    # Act & Assert
    with pytest.raises(KeyError):
        _ = tokenizer.encode(text)


def test_SimpleTokenizerV2_endoftext():
    # Arrange
    expected = [
        1131,
        5,
        355,
        1126,
        628,
        975,
        10,
        1130,
        55,
        988,
        956,
        984,
        722,
        988,
        1131,
        7,
    ]
    vocabulary = the_verdict_vocabulary()
    tokenizer = SimpleTokenizerV2(vocabulary)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join([text1, text2])

    # Act
    ids = tokenizer.encode(text)

    # Assert
    assert ids == expected


def test_SimpleTokenizerV2_encode_decode():
    # Arrange
    expected = "<|unknown|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unknown|>."
    vocabulary = the_verdict_vocabulary()
    tokenizer = SimpleTokenizerV2(vocabulary)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join([text1, text2])

    # Act
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    # Assert
    assert decoded == expected
