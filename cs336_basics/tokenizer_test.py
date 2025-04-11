from cs336_basics.tokenizer import Tokenizer
import pathlib

if __name__ == "__main__":
    prefix = "ts-train"
    OUTPATH = (pathlib.Path(__file__).resolve().parent) / ".." / "out" / "tokenizers" / prefix
    vocab_filepath = OUTPATH / "vocab.txt"
    merges_filepath = OUTPATH / "merges.txt"
    # special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, ["<|endoftext|>"])
    # print(list(tknzr.vocab.items())[:10])

    # test_string = "This is some text to encode."
    # encoded_ids = tokenizer.encode(test_string)
    # print(encoded_ids)
    # decoded_string = tokenizer.decode(encoded_ids)
    # print(decoded_string)
    # assert test_string == decoded_string

    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string
    print(decoded_string)
