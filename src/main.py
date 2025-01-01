def print_preview(file):
    with open(file, "r") as f:
        raw_text = f.read()

        print(f"""Total number of characters: {len(raw_text):,}""")
        print("=========================================================")
        print(raw_text[:99])


def main():
    print_preview("data/the-verdict.txt")


if __name__ == '__main__':
    main()