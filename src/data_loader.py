import tiktoken


def simulate_sliding_window():
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)

    print("=========================================================")
    print(f"""Total number of tokens in the training set: {len(enc_text)}""")

    enc_sample = enc_text[50:]

    print("Visualize sliding window")
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1 : context_size + 1]
    print(f"x: {x}")
    print(f"y:      {y}")

    print("================ T O K E N  I D s ===========================")
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    print("================= A S  W O R D S ============================")
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
