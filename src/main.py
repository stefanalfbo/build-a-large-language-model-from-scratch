import re

def print_preview(file):
    with open(file, "r") as f:
        raw_text = f.read()

        print(f"""Total number of characters: {len(raw_text):,}""")
        print("=========================================================")
        print(raw_text[:99])

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