from src.main import basic_tokenizer

def test_basic_tokenizer():
    # Arrange
    text = "Hello, world. Is this-- a test?"
    expected = ["Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"]

    # Act
    preprocessed = basic_tokenizer(text)

    # Assert
    assert preprocessed == expected