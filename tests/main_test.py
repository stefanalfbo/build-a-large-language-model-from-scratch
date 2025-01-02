from src.main import basic_tokenizer, create_vocabulary

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
        ",": 0, "--": 1, ".": 2, "?": 3, "Hello": 4, "Is": 5, "a": 6, "test": 7, "this": 8, "world": 9
    }

    # Act
    vocabulary = create_vocabulary(preprocessed)

    # Assert
    assert vocabulary == expected