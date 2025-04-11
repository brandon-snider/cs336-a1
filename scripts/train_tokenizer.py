from cs336_basics.train_bpe import train_bpe

# from cs336_basics.train_bpe_cached import train_bpe_cached
import pathlib

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / ".." / "tests" / "fixtures"

if __name__ == "__main__":
    # (vocab, merges) = train_bpe(
    #     input_path="./data/TinyStoriesV2-GPT4-valid.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/tokenizers/ts-valid/merges.txt",
    #     vocab_outpath="./out/tokenizers/ts-valid/vocab.txt",
    # )

    (vocab, merges) = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        merges_outpath="./out/tokenizers/ts-train/merges.txt",
        vocab_outpath="./out/tokenizers/ts-train/vocab.txt",
    )

    # (vocab, merges) = train_bpe(
    #     input_path="./data/owt_valid.txt",
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/tokenizers/owt-valid/merges.txt",
    #     vocab_outpath="./out/tokenizers/owt-valid/vocab.txt",
    # )

    # (vocab, merges) = train_bpe(
    #     input_path="./data/owt_train.txt",
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/tokenizers/owt-train/merges.txt",
    #     vocab_outpath="./out/tokenizers/owt-train/vocab.txt",
    # )

    # (vocab, merges) = train_bpe(
    #     input_path=FIXTURES_PATH / "corpus.en",
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/tokenizers/corpus/merges.txt",
    #     vocab_outpath="./out/tokenizers/corpus/vocab.txt",
    # )

    print("Done.")
