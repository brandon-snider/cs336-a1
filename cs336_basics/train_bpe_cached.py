import os
import heapq
from typing import BinaryIO
import regex as re
import collections
import multiprocessing as mp
import time
import pickle
from functools import reduce

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class ReverseLexOrderPair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: "ReverseLexOrderPair") -> bool:
        return self.pair == other.pair


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        file.seek(pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = pos + found_at
                break
            pos += mini_chunk_size

    return sorted(set(chunk_boundaries))


def merge_freq_dicts(dict1: dict[tuple[bytes], int], dict2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result


def get_pair_freqs(
    freqs: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set]]:
    pair_freqs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set] = collections.defaultdict(set)

    for symbols, freq in freqs.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freqs[pair] += freq
            pairs_to_keys[pair].add(symbols)

    return pair_freqs, pairs_to_keys


def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    new_symbols = []
    i = 0
    while i < len(old_repr):
        if i < len(old_repr) - 1 and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbols.append(old_repr[i] + old_repr[i + 1])
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1
    return tuple(new_symbols)


def merge(
    freqs: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set],
    pair: tuple[bytes, bytes],
) -> set:
    changed_pairs = set()
    keys_to_modify = pairs_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = freqs.pop(old_key)
        new_key = build_new_repr(old_key, pair)

        # Decrement old adjacencies
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            pair_freqs[left, right] -= old_freq
            changed_pairs.add((left, right))
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left, right].discard(old_key)

        # Increment new adjacencies
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            pair_freqs[left, right] += old_freq
            changed_pairs.add((left, right))
            pairs_to_keys[left, right].add(new_key)

        freqs[new_key] = freqs.get(new_key, 0) + old_freq

    pairs_to_keys[pair] = set()
    return changed_pairs


def write_merges(merges, outpath):
    with open(outpath, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved {len(merges)} merges to {outpath}")


def write_vocab(vocab, outpath):
    with open(outpath, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary with {len(vocab)} tokens to {outpath}")


def pretokenize_chunk_with_cache(
    chunk_str: str,
    special_pattern: re.Pattern,
    local_cache: dict[str, tuple[bytes]],
    global_cache: dict[str, tuple[bytes]],
    freqs: dict[tuple[bytes], int],
    sync_interval: int,
    iteration_counter: list[int],  # single-element list to let us modify from inside
) -> None:
    """
    Tokenize a single chunk string, using local_cache and occasionally merging into global_cache.
    Updates `freqs` in-place. The `iteration_counter` is a mutable int-like container
    so we can keep track of how many tokens we've processed, to trigger syncs at intervals.
    """
    sub_chunks = [chunk_str]
    if special_pattern:
        sub_chunks = special_pattern.split(chunk_str)

    for sub_chunk in sub_chunks:
        for match in PAT.finditer(sub_chunk):
            token_str = match.group()

            # Local cache first
            if token_str in local_cache:
                match_bytes = local_cache[token_str]
            else:
                # Check global cache
                if token_str in global_cache:
                    match_bytes = global_cache[token_str]
                    # Cache it locally too
                    local_cache[token_str] = match_bytes
                else:
                    # Construct new representation
                    match_bytes = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                    local_cache[token_str] = match_bytes

            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1

            # Periodically merge local → global
            iteration_counter[0] += 1
            if iteration_counter[0] % sync_interval == 0:
                # Merge local_cache into global_cache
                for k, v in local_cache.items():
                    global_cache[k] = v


def batch_pre_tokenize(
    chunk_strs: list[str],
    special_regex: str,
    global_cache: dict[str, tuple[bytes]],
    sync_interval: int,
) -> dict[tuple[bytes], int]:
    """
    Tokenizes a batch of chunks, maintaining its own local cache but periodically
    syncing back into global_cache. Returns the aggregated frequency dictionary.
    """
    local_cache: dict[str, tuple[bytes]] = {}
    special_pattern = re.compile(special_regex) if special_regex else None
    freqs: dict[tuple[bytes], int] = {}
    iteration_counter = [0]  # single-element list to mutate inside the function

    for chunk_str in chunk_strs:
        pretokenize_chunk_with_cache(
            chunk_str,
            special_pattern,
            local_cache,
            global_cache,
            freqs,
            sync_interval,
            iteration_counter,
        )

    # Final sync at end of batch
    for k, v in local_cache.items():
        global_cache[k] = v

    return freqs


def pre_tokenize(
    input_path: str, special_tokens: list[str], sync_interval: int = 20000, batch_size: int = 8
) -> dict[tuple[bytes], int]:
    """
    Splits a file into chunk boundaries aligned with a special token, then
    processes them in parallel. Each worker processes a batch of chunks and
    periodically merges its local cache of (string→tuple-of-bytes) into a
    global manager dict. Returns aggregated frequency dictionary.
    """
    # Shared manager dictionary for token cache
    manager = mp.Manager()
    global_token_cache = manager.dict()

    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    special_pattern_str = "|".join(re.escape(tok) for tok in special_tokens) if special_tokens else ""
    all_chunks = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
            all_chunks.append(chunk_str)

    # Divide chunk strings into batches
    chunk_batches = [all_chunks[i : i + batch_size] for i in range(0, len(all_chunks), batch_size)]

    # Map each batch to a worker
    async_results = []
    for batch in chunk_batches:
        async_results.append(
            pool.apply_async(
                batch_pre_tokenize,
                (
                    batch,  # chunk_strs
                    special_pattern_str,
                    global_token_cache,
                    sync_interval,
                ),
            )
        )

    pool.close()
    pool.join()

    # Merge all partial freq dicts
    freq_dicts = [res.get() for res in async_results]
    combined_freqs = reduce(merge_freq_dicts, freq_dicts, {})

    return combined_freqs


def train_bpe_cached(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    merges_outpath: str = None,
    vocab_outpath: str = None,
    sync_interval: int = 1000000,
    batch_size: int = 8,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    train_start_time = time.time()

    # Initialize special tokens + single-byte tokens
    initial_tokens = [tok.encode("UTF-8") for tok in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    merges = []

    # Pre-tokenize with multi-process caching
    print("Pre-tokenize: start")
    start_time = time.time()
    freqs = pre_tokenize(input_path, special_tokens, sync_interval, batch_size)
    print(f"Pre-tokenize: done in {time.time() - start_time:.2f}s")

    # Build pair freqs
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)
    pair_heap = []
    for p, f in pair_freqs.items():
        if f > 0:
            heapq.heappush(pair_heap, (-f, ReverseLexOrderPair(p), p))

    n_initial_tokens = len(initial_tokens)
    n_merges = vocab_size - n_initial_tokens

    # Merge loop
    for i in range(n_initial_tokens, n_initial_tokens + n_merges):
        if not pair_heap:
            break

        while pair_heap:
            neg_freq, _, top_pair = heapq.heappop(pair_heap)
            freq = -neg_freq
            if pair_freqs.get(top_pair, 0) == freq:
                pair = top_pair
                break
            if top_pair in pair_freqs and pair_freqs[top_pair] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[top_pair], ReverseLexOrderPair(top_pair), top_pair))
        else:
            break

        if pair_freqs.get(pair, 0) <= 0:
            break

        vocab[i] = pair[0] + pair[1]
        merges.append(pair)

        changed_pairs = merge(freqs, pair_freqs, pairs_to_keys, pair)
        for cp in changed_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[cp], ReverseLexOrderPair(cp), cp))

        if (i - n_initial_tokens + 1) % 100 == 0 or (i == n_initial_tokens + n_merges - 1):
            print(f"{i - n_initial_tokens + 1}/{n_merges} merges completed")

    print(f"Training finished in {time.time() - train_start_time:.2f}s")

    # Optionally save merges and vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe_cached(
        input_path="./data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        merges_outpath="./out/ts-valid-merges-2.txt",
        vocab_outpath="./out/ts-valid-vocab-2.txt",
        sync_interval=50000,  # merge local→global every N tokens
        batch_size=4096,  # each worker handles 8 small chunks per task
    )
