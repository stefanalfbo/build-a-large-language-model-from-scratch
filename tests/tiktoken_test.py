import tiktoken


def test_tiktoken():
    # Arrange
    expected_strings = "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
    expected_ids = [
        15496,
        11,
        466,
        345,
        588,
        8887,
        30,
        220,
        50256,  # <|endoftext|>
        554,
        262,
        4252,
        18250,
        8812,
        2114,
        1659,
        617,
        34680,
        27271,
        13,
    ]
    tokenizer = tiktoken.get_encoding("gpt2")
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )

    # Act
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    strings = tokenizer.decode(ids)

    # Assert
    assert ids == expected_ids
    assert strings == expected_strings


def test_unknown_token():
    # Arrange
    unknown_words = "Akwirw ier"
    tokenizer = tiktoken.get_encoding("gpt2")

    # Act
    ids = tokenizer.encode(unknown_words)
    text = tokenizer.decode(ids)

    # Assert
    assert ids == [33901, 86, 343, 86, 220, 959]
    assert tokenizer.decode([33901]) == "Ak"
    assert tokenizer.decode([86]) == "w"
    assert tokenizer.decode([343]) == "ir"
    assert tokenizer.decode([86]) == "w"
    assert tokenizer.decode([220]) == " "
    assert tokenizer.decode([959]) == "ier"
    assert text == unknown_words
