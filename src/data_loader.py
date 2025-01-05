import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


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


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def embedding_vector_conversion(raw_text, vocabulary, output_dimensions):
    token_embedding_layer = torch.nn.Embedding(len(vocabulary), output_dimensions)

    max_length = 4

    # Create a dataloader and sample each batch
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, _ = next(data_iter)

    # Use the embedding layer to convert token ids to embedding vectors
    token_embeddings = token_embedding_layer(inputs)

    # For a GPT model’s absolute embedding approach, we just need
    # to create another embedding layer that has the same embedding
    # dimension as the token_embedding_layer.

    # The context_length is a variable that represents the supported
    # input size of the LLM
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dimensions)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    # The positional embedding tensor consists of four 256-dimensional
    # vectors. We can now add these directly to the token embeddings
    input_embeddings = token_embeddings + pos_embeddings

    return input_embeddings
