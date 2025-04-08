import regex as re
import collections
import multiprocessing as mp
import time
import pickle
from functools import reduce


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CHUNK_SIZE = 1024 * 1024  # 1MB chunks


def pre_tokenize_chunk(chunk: str) -> dict[tuple[bytes], int]:
    """Process a chunk of text and return frequency dictionary"""
    freqs: dict[tuple[bytes], int] = {}

    docs = chunk.split("<|endoftext|>")

    for i, doc_text in enumerate(docs):
        for match in re.finditer(PAT, doc_text):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1

    return freqs


def merge_freq_dicts(dict1: dict[tuple[bytes], int], dict2: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    """Merge two frequency dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result


def pre_tokenize(input_path: str) -> dict[tuple[bytes], int]:
    """Get the initial coarse-grained frequencies using regex with multiprocessing"""

    chunk_freqs = []
    pool = mp.Pool(processes=mp.cpu_count())

    with open(input_path) as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            # Process each chunk immediately and store only the result
            chunk_freqs.append(pool.apply_async(pre_tokenize_chunk, (chunk,)))

    pool.close()
    pool.join()

    # Get results from async objects
    freq_dicts = [async_result.get() for async_result in chunk_freqs]

    combined_freqs = reduce(merge_freq_dicts, freq_dicts)
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
    initial_tokens = [token.encode("UTF-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    vocab = {i: token for i, token in enumerate(initial_tokens)}
    merges = []

    print("Pre-tokenize: start")
    start_time = time.time()
    # Coarse-grained frequencies
    # e.g. {(b'l', b'o',b 'w', b'e', b'r'): 12, (b'h', b'i',b'g', b'h'): 3, ...}
    freqs = pre_tokenize(input_path)
    print(f"Pre-tokenize: finished in {time.time() - start_time:.2f}s")

    print("Initial pair frequencies: start")
    start_time = time.time()
    # Initial pair frequencies, e.g. {(b'a', b'b'): 2, (b'c, b'de'): 12, ...}
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)
    print(f"Initial pair frequencies: finished in {time.time() - start_time:.2f}s")

    n_initial_tokens = len(initial_tokens)
    n_merges = vocab_size - n_initial_tokens

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
        if i > n_initial_tokens and (i - n_initial_tokens) % 100 == 0:
            print(f"{i - n_initial_tokens}/{n_merges} merges completed (runtime: {time.time() - start_time:.2f}s)")

    # Optionally write merges and vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return (vocab, merges)
