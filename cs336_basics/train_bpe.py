import os
import heapq
from typing import BinaryIO
import regex as re
import collections
import multiprocessing as mp
import time
import pickle
from functools import reduce

# Regex for coarse tokenization
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class ReverseLexOrderPair:
    """
    Encapsulates (bytes, bytes) so that in a min-heap, the "largest in normal lex order"
    is treated as the smallest. Ensures that tie frequencies pop in reverse lex order.
    """

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        # Invert normal order: self < other if self is > other (so larger lex sorts first).
        return self.pair > other.pair

    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Find chunk boundaries by reading forward from guessed positions
    until split_special_token is found (or EOF). Ensures alignment.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks

    # Initial boundary guesses (uniformly spaced); force last boundary at EOF
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        file.seek(pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                # If EOF is reached before finding split token
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # Found the split token; adjust boundary precisely
                chunk_boundaries[bi] = pos + found_at
                break
            pos += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(chunk: str, special_pattern: re.Pattern | None) -> dict[tuple[bytes], int]:
    """Regex tokenizes the chunk. Splits first on special tokens, then uses PAT."""
    freqs: dict[tuple[bytes], int] = {}
    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk]

    for sub_chunk in sub_chunks:
        for match in PAT.finditer(sub_chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1

    return freqs


def merge_freq_dicts(dict1: dict[tuple[bytes], int], dict2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    """Adds frequencies from dict2 into dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result


def pre_tokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """
    Splits a file into chunks aligned with <|endoftext|>, then tokenizes each chunk
    in parallel. Returns aggregated frequency dict.
    """
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    chunk_freqs = []
    special_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens)) if special_tokens else None

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Read each chunk in bytes, decode, then apply_async for parallel tokenization
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
            chunk_freqs.append(pool.apply_async(pre_tokenize_chunk, (chunk_str, special_pattern)))

    pool.close()
    pool.join()

    # Collect and merge partial results
    freq_dicts = [res.get() for res in chunk_freqs]
    combined_freqs = reduce(merge_freq_dicts, freq_dicts, {})
    return combined_freqs


def get_pair_freqs(
    freqs: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """
    Builds a pair-frequency table and reverse mapping (pair -> set of keys).
    """
    pair_freqs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = collections.defaultdict(set)

    for symbols, freq in freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freqs[pair] += freq
            pairs_to_keys[pair].add(symbols)

    return pair_freqs, pairs_to_keys


def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    """Replaces every occurrence of pair=(x,y) in old_repr with the merged symbol x+y."""
    new_symbols = []
    i = 0
    while i < len(old_repr):
        if i < len(old_repr) - 1 and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbols.append(old_repr[i] + old_repr[i + 1])  # merges, e.g. b'A' + b'B' => b'AB'
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1
    return tuple(new_symbols)


def merge(
    freqs: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
    pair: tuple[bytes, bytes],
) -> set[tuple[bytes, bytes]]:
    """Merges 'pair' into freqs and updates pair_freqs & pairs_to_keys for all affected old/new keys."""
    changed_pairs = set()
    keys_to_modify = pairs_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = freqs.pop(old_key)
        new_key = build_new_repr(old_key, pair)

        # Decrement frequencies in pair_freqs for old_key's adjacencies
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            pair_freqs[left, right] -= old_freq
            changed_pairs.add((left, right))
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left, right].discard(old_key)

        # Increment frequencies for new_key's adjacencies
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            pair_freqs[left, right] += old_freq
            changed_pairs.add((left, right))
            pairs_to_keys[left, right].add(new_key)

        # Put new_key back with updated freq
        freqs[new_key] = freqs.get(new_key, 0) + old_freq

    pairs_to_keys[pair] = set()
    return changed_pairs


def write_merges(merges, outpath):
    """Pickle the merges list to a binary file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved {len(merges)} merges to {outpath}")


def write_vocab(vocab, outpath):
    """Pickle the vocab dict to a binary file."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary with {len(vocab)} tokens to {outpath}")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    merges_outpath: str = None,
    vocab_outpath: str = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains byte-level BPE on a text file, returning:
      - vocab: dict[int, bytes]
      - merges: list of merged pairs
    """
    train_start_time = time.time()

    # Initialize special tokens and single-byte tokens
    initial_tokens = [tok.encode("UTF-8") for tok in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    merges = []

    print("Pre-tokenize: start")
    start_time = time.time()
    freqs = pre_tokenize(input_path, special_tokens)
    print(f"Pre-tokenize: finished in {time.time() - start_time:.2f}s")

    print("Initial pair frequencies: start")
    start_time = time.time()
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)

    # Build a max-heap by pushing negative frequencies
    pair_heap = []
    for p, f in pair_freqs.items():
        if f > 0:
            heapq.heappush(pair_heap, (-f, ReverseLexOrderPair(p), p))

    print(f"Initial pair frequencies: finished in {time.time() - start_time:.2f}s")

    n_initial_tokens = len(initial_tokens)
    n_merges = vocab_size - n_initial_tokens

    print("Merge: start")
    start_time = time.time()

    for i in range(n_initial_tokens, n_initial_tokens + n_merges):
        if not pair_heap:
            break

        # Pop until we find the top pair that still matches pair_freqs
        while pair_heap:
            neg_freq, _, top_pair = heapq.heappop(pair_heap)
            freq = -neg_freq
            if pair_freqs.get(top_pair, 0) == freq:
                pair = top_pair
                break
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[top_pair], ReverseLexOrderPair(top_pair), top_pair))
        else:
            # If pair_heap is empty after the loop, we are done
            break

        if pair_freqs.get(pair, 0) <= 0:
            break

        # Add this new merge token to vocab and record the merge
        vocab[i] = pair[0] + pair[1]
        merges.append(pair)

        # Merge in freqs, then update the heap for pairs changed by this merge
        changed_pairs = merge(freqs, pair_freqs, pairs_to_keys, pair)
        for cp in changed_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[cp], ReverseLexOrderPair(cp), cp))

        # Print progress every 100 merges or at the last iteration
        if ((i > n_initial_tokens) and ((i - n_initial_tokens + 1) % 100 == 0)) or (
            i == n_initial_tokens + n_merges - 1
        ):
            print(
                f"{i - n_initial_tokens + 1}/{n_merges} merges completed (merge runtime: {time.time() - start_time:.2f}s)"
            )

    print(f"Merges completed in {time.time() - start_time:.2f}s")
    print(f"Training completed in {time.time() - train_start_time:.2f}s")

    # Optionally save merges and vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return vocab, merges


if __name__ == "__main__":
    (vocab, merges) = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        merges_outpath="./out/ts-valid-merges-2.txt",
        vocab_outpath="./out/ts-valid-vocab-2.txt",
    )
