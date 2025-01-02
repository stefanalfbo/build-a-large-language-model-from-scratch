import re

def print_preview(file):
    with open(file, "r") as f:
        raw_text = f.read()
        preprocessed = basic_tokenizer(raw_text)
        vocabulary = create_vocabulary(preprocessed)

        print(f"""Total number of characters: {len(raw_text):,}""")
        print("=========================================================")
        print(f"""Preview: {raw_text[:99]}""")
        print()
        print(f"""Total number of tokens: {len(preprocessed):,}""")
        print("=========================================================")
        print(f"""Preview: {preprocessed[:30]}""")
        print()
        print(f"""Vocabulary size: {len(vocabulary):,}""")
        print("=========================================================")
        print(f"""Preview: {list(vocabulary.items())[:10]}""")

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

def main():
    print_preview("data/the-verdict.txt")


if __name__ == '__main__':
    main()