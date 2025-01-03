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
