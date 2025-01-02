import re

def print_preview(file):
    with open(file, "r") as f:
        raw_text = f.read()
        preprocessed = basic_tokenizer(raw_text)

        print(f"""Total number of characters: {len(raw_text):,}""")
        print("=========================================================")
        print(f"""Preview: {raw_text[:99]}""")
        print()
        print(f"""Total number of tokens: {len(preprocessed):,}""")
        print("=========================================================")
        print(f"""Preview: {preprocessed[:30]}""")

def basic_tokenizer(text: str) -> list[str]:
    """Simple tokenizer that splits text into tokens. 
    
    Args: 
        text (str): The text to tokenize.
    
    Returns:
        list[str]: A list of whitespace-free tokens.
    """
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)

    return [token for token in result if token.strip()]

def main():
    print_preview("data/the-verdict.txt")


if __name__ == '__main__':
    main()