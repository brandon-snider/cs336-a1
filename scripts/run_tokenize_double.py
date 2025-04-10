import os
import pathlib
import numpy as np
import time
from cs336_basics.tokenizer import Tokenizer


def tokenize(vocab_path, merges_path, data_path, output_path):
    # Instantiate the tokenizer
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

    # 0: use compression ratio to estimate total token count
    sample_size = 250_000
    file_size = os.path.getsize(data_path)
    with open(data_path, "rb") as f:
        sample_bytes = f.read(sample_size)
    sample_str = sample_bytes.decode("utf-8", errors="replace")
    sample_tokens = len(tokenizer.encode(sample_str))
    ratio = len(sample_bytes) / sample_tokens
    approx_total_tokens = int(file_size / ratio)

    progress_interval = 1_000_000

    print(f"Approximately {approx_total_tokens:,} tokens in the dataset")
    print("-" * 100)

    # First pass: count total tokens
    first_pass_start = time.time()
    total_tokens = 0
    with open(data_path) as f:
        for _ in tokenizer.encode_iterable(f):
            total_tokens += 1

            if total_tokens % progress_interval == 0:
                seconds_elapsed = time.time() - first_pass_start
                tok_per_second = total_tokens / seconds_elapsed
                seconds_remaining = (approx_total_tokens - total_tokens) / tok_per_second
                print(
                    f"[Counting] {total_tokens:,} / {approx_total_tokens:,} tok counted | ~{total_tokens / approx_total_tokens:.2%} | {int(tok_per_second / 1000):,}k tok/s | {seconds_elapsed:.2f}s elapsed | {seconds_remaining:.2f}s remaining"
                )

    print("-" * 100)
    print(
        f"Counted {total_tokens:,} tokens in {time.time() - first_pass_start:.2f}s (Pre-count estimate: {approx_total_tokens:,})"
    )
    print("-" * 100)

    # Create a memory-mapped array (raw binary file, not .npy)
    mm = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

    # Second pass: fill the array with token IDs
    second_pass_start = time.time()
    idx = 0
    with open(data_path) as f:
        for token_id in tokenizer.encode_iterable(f):
            mm[idx] = token_id
            idx += 1

            if idx % progress_interval == 0:
                seconds_elapsed = time.time() - second_pass_start
                tok_per_second = idx / seconds_elapsed
                seconds_remaining = (total_tokens - idx) / tok_per_second
                print(
                    f"[Writing] {idx:,} / {total_tokens:,} tok written | ~{idx / total_tokens:.2%} | {int(tok_per_second / 1000):,}k tok/s | {seconds_elapsed:.2f}s elapsed | {seconds_remaining:.2f}s remaining"
                )

    print("-" * 100)
    print(f"Wrote {total_tokens:,} tokens in {time.time() - second_pass_start:.2f}s")

    mm.flush()


if __name__ == "__main__":
    BASE_PATH = pathlib.Path(__file__).resolve().parent / ".."

    vocab_path, merges_path = BASE_PATH / "out/ts-train-vocab.txt", BASE_PATH / "out/ts-train-merges.txt"
    data_path = BASE_PATH / "data/TinyStoriesV2-GPT4-train.txt"
    output_path = BASE_PATH / "out/ts-train-tokens.bin"

    # vocab_path, merges_path = BASE_PATH / "out/owt-train-vocab.txt", BASE_PATH / "out/owt-train-merges.txt"
    # data_path = BASE_PATH / "data/owt_valid.txt"
    # output_path = BASE_PATH / "out/owt-valid-tokens.bin"

    tokenize(vocab_path, merges_path, data_path, output_path)
