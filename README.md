# Build a Large Language Model from scratch

This repository is based on the material from the book [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by [Sebastian Raschka](https://sebastianraschka.com/). The book is published by Manning Publications.

## Getting started

Start with cloning the repository, `git clone https://github.com/stefanalfbo/build-a-large-language-model-from-scratch.git`.

Then, install the required packages with `make sync`.

## References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The paper that introduced the transformer architecture.
* [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159) - A comparably large dataset for language model pre-training.
* [Improving Language Understanding
by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - The paper that introduced the GPT model.
* [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - A paper that introduces a method to train language models with human feedback.

## Dependencies

* [tiktoken](https://github.com/openai/tiktoken) - which is a fast BPE (byte pair encoding) tokenizer.
