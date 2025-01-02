import re


def basic_tokenizer(text: str) -> list[str]:
    """Simple tokenizer that splits text into tokens.

    Args:
        text (str): The text to tokenize.

    Returns:
        list[str]: A list of whitespace-free tokens.
    """
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)

    return [token for token in result if token.strip()]


def create_vocabulary(preprocessed: list[str]) -> list[str]:
    all_tokens = sorted(set(preprocessed))

    return {token: i for i, token in enumerate(all_tokens)}


def the_verdict_vocabulary():
    with open("data/the-verdict.txt", "r") as f:
        raw_text = f.read()
        preprocessed = basic_tokenizer(raw_text)
        return create_vocabulary(preprocessed)


class SimpleTokenizerV1:
    """Tokenizes and detokenizes a string using a simple vocabulary."""

    def __init__(self, vocabulary: dict[str, int]):
        # For simple access in the encode and decode methods
        self.str_to_int = vocabulary
        # The inverse of the vocabulary
        self.int_to_str = {i: s for s, i in vocabulary.items()}

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of token IDs."""
        result = re.split(r'([,.?_!"()\']|--|\s)', text)

        preprocessed = [token for token in result if token.strip()]

        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs into a string."""
        text = " ".join([self.int_to_str[i] for i in ids])

        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)
