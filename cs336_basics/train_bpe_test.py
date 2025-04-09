from cs336_basics.train_bpe import train_bpe
import pathlib

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / ".." / "tests" / "fixtures"

if __name__ == "__main__":
    # (vocab, merges) = train_bpe(
    #     input_path="./data/TinyStoriesV2-GPT4-valid.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/ts-valid-merges.txt",
    #     vocab_outpath="./out/ts-valid-vocab.txt",
    # )

    # (vocab, merges) = train_bpe(
    #     input_path="./data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/ts-train-merges.txt",
    #     vocab_outpath="./out/ts-train-vocab.txt",
    # )

    # (vocab, merges) = train_bpe(
    #     input_path="./data/owt_valid.txt",
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/owt-valid-merges.txt",
    #     vocab_outpath="./out/owt-valid-vocab.txt",
    # )

    (vocab, merges) = train_bpe(
        input_path="./data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
        merges_outpath="./out/owt-train-merges.txt",
        vocab_outpath="./out/owt-train-vocab.txt",
    )

    # (vocab, merges) = train_bpe(
    #     input_path=FIXTURES_PATH / "corpus.en",
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    #     merges_outpath="./out/corpus-merges.txt",
    #     vocab_outpath="./out/corpus-vocab.txt",
    # )

    print("Done.")
