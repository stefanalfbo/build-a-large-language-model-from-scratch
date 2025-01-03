from importlib.metadata import version

from simple_tokenizer import basic_tokenizer, create_vocabulary


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
        print(f"""Preview, first 5: {list(vocabulary.items())[:5]}""")
        print(f"""Preview, last 5: {list(vocabulary.items())[-5:]}""")


def print_tiktoken_version():
    print("=========================================================")
    print(f"""tiktoken version: {version("tiktoken")}""")
    print("=========================================================")
    print()


def main():
    print_tiktoken_version()
    print_preview("data/the-verdict.txt")


if __name__ == "__main__":
    main()
