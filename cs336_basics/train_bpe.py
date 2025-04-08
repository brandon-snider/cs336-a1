import os
from typing import BinaryIO
import regex as re
import collections
import multiprocessing as mp
import time
import pickle
from functools import reduce


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [
        i * chunk_size for i in range(desired_num_chunks + 1)
    ]  # Chunks start on previous index, don't include last index
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            if mini_chunk == b"":  # If EOF, this boundary should be at the end of the file
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)  # Find the special token in the mini chunk
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """Process a chunk of text and return frequency dictionary"""
    freqs: dict[tuple[bytes], int] = {}

    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        sub_chunks = re.split(pattern, chunk)
    else:
        sub_chunks = [chunk]

    for sub_chunk in sub_chunks:
        for match in re.finditer(PAT, sub_chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1

    return freqs


def merge_freq_dicts(dict1: dict[tuple[bytes], int], dict2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    """Merge two frequency dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result


def pre_tokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """Get the initial coarse-grained frequencies using regex with multiprocessing,
    ensuring chunk boundaries align with '<|endoftext|>' tokens.
    """
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    chunk_freqs = []

    with open(input_path, "rb") as f:
        # Find boundary offsets
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # For each boundary pair, read that chunk and decode to str
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # Seek to the start offset
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

            #  Distribute work to each worker process
            chunk_freqs.append(
                pool.apply_async(
                    pre_tokenize_chunk,
                    (chunk_str, special_tokens),
                )
            )

    pool.close()
    pool.join()

    freq_dicts = [async_result.get() for async_result in chunk_freqs]

    # Combine results
    combined_freqs = reduce(merge_freq_dicts, freq_dicts, {})
    return combined_freqs


def get_pair_freqs(
    freqs: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """Get the initial pair-frequency table from the coarse-grained frequencies table"""
    pair_freqs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = collections.defaultdict(set)

    for symbols, freq in freqs.items():
        for i in range(len(symbols) - 1):
            pair_freqs[symbols[i], symbols[i + 1]] += freq
            pairs_to_keys[symbols[i], symbols[i + 1]].add(symbols)

    return pair_freqs, pairs_to_keys


def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    new_symbols = []
    i = 0
    while i < len(old_repr):
        # Merge the pair whenever we see it
        if i < len(old_repr) - 1 and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbols.append(old_repr[i] + old_repr[i + 1])  # b'A' + b'B' => b'AB'
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1

    new_repr = tuple(new_symbols)
    return new_repr


def merge(
    freqs: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
    pair: tuple[bytes, bytes],
) -> None:
    """Merge a pair in the coarse-grained token frequency table, and
    synchronize pair_freqs and pairs_to_keys for pairs affected by the merge."""
    keys_to_modify = pairs_to_keys[pair].copy()

    for old_key in keys_to_modify:
        old_freq = freqs.pop(old_key)
        new_key = build_new_repr(old_key, pair)

        # 1. Decrement pair_freqs for existing adjacencies in old_key
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            pair_freqs[left, right] -= old_freq
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left, right].discard(old_key)

        # 3. Increment pair_freqs for new_key's adjacencies
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            pair_freqs[left, right] += old_freq
            pairs_to_keys[left, right].add(new_key)

        # 4. Put the new_key back into freqs
        freqs[new_key] = freqs.get(new_key, 0) + old_freq

    pairs_to_keys[pair] = set()


def write_merges(merges, outpath):
    """Write merges directly to a binary file using pickle"""
    with open(outpath, "wb") as f:
        pickle.dump(merges, f)

    print(f"Saved {len(merges)} merges to {outpath}")


def write_vocab(vocab, outpath):
    """Write vocab directly to a binary file using pickle"""
    import pickle

    with open(outpath, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Saved vocabulary with {len(vocab)} tokens to {outpath}")


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], merges_outpath: str = None, vocab_outpath: str = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains a byte-level BPE tokenizer with the specified vocab_size on the
    text in the file at input_path.

    Returns:
    - vocab: dict[int,bytes]
    - merges: list[tuple[bytes,bytes]]
    """
    train_start_time = time.time()

    initial_tokens = [token.encode("UTF-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    merges = []

    print("Pre-tokenize: start")
    start_time = time.time()
    # Coarse-grained token frequencies e.g. {(b'l', b'o', b'w'): 12, (b'h', b'i',b'g', b'h'): 3, ...}
    freqs = pre_tokenize(input_path, special_tokens)
    print(f"Pre-tokenize: finished in {time.time() - start_time:.2f}s")

    print("Initial pair frequencies: start")
    start_time = time.time()
    # Pair frequencies, e.g. {(b'a', b'b'): 2, (b'c, b'de'): 12, ...}
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)
    print(f"Initial pair frequencies: finished in {time.time() - start_time:.2f}s")

    n_initial_tokens = len(initial_tokens)
    n_merges = vocab_size - n_initial_tokens

    print("Merge: start")
    start_time = time.time()

    for i in range(n_initial_tokens, n_initial_tokens + n_merges):
        # If max. vocab size is larger than largest possible vocab for provided data
        if not pair_freqs:
            break

        # Most frequent pair, e.g. (b'a', b'be')
        best_pair = max(pair_freqs, key=lambda x: (pair_freqs[x], x))

        if pair_freqs[best_pair] <= 0:
            # No more merges that improve anything
            break

        # Create new vocab entry and note the merge
        vocab[i] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # Merge the pair in freqs, and update pair_freqs incrementally
        merge(freqs, pair_freqs, pairs_to_keys, best_pair)

        # Print progress every 10 merges
        if (i > n_initial_tokens and (i - n_initial_tokens + 1) % 100 == 0) or i == n_initial_tokens + n_merges - 1:
            print(
                f"{i - n_initial_tokens + 1}/{n_merges} merges completed (merge runtime: {time.time() - start_time:.2f}s)"
            )

    print(f"Merges completed in {time.time() - start_time:.2f}s")
    print(f"Training completed in {time.time() - train_start_time:.2f}s")

    # Optionally write merges and vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return (vocab, merges)
