import regex as re
import collections

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENC = "UTF-8"


def pre_tokenize(input_path: str) -> dict[tuple[bytes], int]:
    """Get the initial coarse-grained frequencies using regex"""

    with open(input_path) as f:
        contents = f.read()

    freqs: dict[tuple[bytes], int] = {}

    for match in re.finditer(PAT, contents):
        match_str = match.group()
        match_bytes = tuple(c.encode(ENC) for c in match_str)

        if match_bytes not in freqs:
            freqs[match_bytes] = 0

        freqs[match_bytes] += 1

    return freqs


def get_pair_freqs(freqs: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    """Get the frequencies of each byte pair from the coarse-grained frequencies table"""

    pairs: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)

    for symbols, freq in freqs.items():
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq

    return pairs


def merge(freqs: dict[tuple[bytes], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
    """Merge the pair in the table of coarse-grained frequencies

    Example: {(b'i', b'b', b'c'): 5} -> {(b'i', b'bc'): 5}
    """

    new_freqs = {}

    for symbols, freq in freqs.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                # Merge the pair
                new_symbols.append(pair[0] + pair[1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1

        new_freqs[tuple(new_symbols)] = new_freqs.get(tuple(new_symbols), 0) + freq

    return new_freqs


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains a byte-level BPE tokenizer with the specified `vocab_size` on the
    text in the file at `input_path.

    Returns:
    - vocab: dict[int,bytes]
    - merges: list[tuple[bytes,bytes]]
    """
    initial_tokens = special_tokens + [chr(i) for i in range(256)]
    vocab = {token: i for i, token in enumerate(initial_tokens)}
    merges = []

    # Coarse-grained frequencies,
    # e.g. {(b'l', b'o',b 'w', b'e', b'r'): 12, (b'h', b'i',b'g', b'h'): 3, ...}
    freqs = pre_tokenize(input_path)

    n_initial_tokens = len(initial_tokens)
    n_merges = vocab_size - n_initial_tokens

    for i in range(n_initial_tokens, n_initial_tokens + n_merges):
        # Pair frequencies, e.g. {(b'a', b'b'): 2, (b'c, b'de'): 12, ...}
        pairs = get_pair_freqs(freqs)

        # If max. vocab size is larger than largest possible vocab for provided data
        if not pairs:
            break

        # Most frequent pair, e.g. (b'a', b'be')
        best = max(pairs, key=lambda x: (pairs.get(x), x))

        # Create new vocab entry from most frequent pair
        vocab[best[0] + best[1]] = i

        # Replace individual tokens with pair in coarse-grained frequencies
        freqs = merge(freqs, best)

        # Append to merges
        merges.append(best)

    return (vocab, merges)
