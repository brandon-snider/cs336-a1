from collections.abc import Iterable, Iterator
from collections import defaultdict
import regex as re
import pickle
import heapq

from cs336_basics import train_bpe


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """Constructs a tokenizer from a vocab, list of merges, and (optionally) list of special tokens"""
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}

        # Sort special tokens by length in descending order to prioritize longer matches
        # Special tokens are not encoded here because the strings are used to construct a regex in `encode`
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

        # Add special tokens to vocab and vocab_inv with unique token ids
        if special_tokens:
            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                # Encode the tokens to match the type of other vocab entries
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Constructs and returns a Tokenizer from a serialized vocab, list of merges, and (optionally) list
        of special tokens"""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text: str) -> list[int]:
        """Encodes an input text into a sequence of token IDs."""
        pretokens, pretoken_reprs, pairs_to_pretokens = self._pretokenize(text)

        heap = []
        for pair in pairs_to_pretokens:
            if pair in self.merges_dict:
                priority = self.merges_dict[pair]
                heapq.heappush(heap, (priority, pair))

        while heap:
            priority, pair = heapq.heappop(heap)

            # If pair is no longer valid or doesn't exist in merges_dict, skip
            if pair not in pairs_to_pretokens or pair not in self.merges_dict:
                continue

            # Have to copy to prevent set size changing during iteration
            pretokens_to_update = pairs_to_pretokens[pair].copy()

            # Iterate over the pretokens whose representations contain the merged bytes
            for pretoken in pretokens_to_update:
                old_repr = pretoken_reprs[pretoken]
                new_repr = train_bpe.build_new_repr(old_repr, pair)

                """Updates the map from pairs to the set of pretokens in which they appear, after a merge."""
                for i in range(len(old_repr) - 1):
                    left, right = old_repr[i], old_repr[i + 1]
                    pairs_to_pretokens[left, right].discard(pretoken)
                    if len(pairs_to_pretokens[left, right]) == 0:
                        del pairs_to_pretokens[left, right]

                for i in range(len(new_repr) - 1):
                    left, right = new_repr[i], new_repr[i + 1]
                    pairs_to_pretokens[left, right].add(pretoken)
                    new_pair = (new_repr[i], new_repr[i + 1])
                    if new_pair in self.merges_dict:
                        heapq.heappush(heap, (self.merges_dict[new_pair], new_pair))

                pretoken_reprs[pretoken] = new_repr

        ids = [self.vocab_inv[token] for pretoken in pretokens for token in pretoken_reprs[pretoken]]
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_chunk(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []

        for part in special_chunks:
            if part in self.special_tokens:
                # this is a special token, encode it separately as a special case
                ids.append(self.vocab_inv[part.encode("UTF-8")])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self._encode_chunk(part))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), returns a generator that lazily
        yields token IDs. Required for memory-eﬀicient tokenization of large files."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decodes a sequence of token IDs into text."""
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")

    def _pretokenize(self, text: str) -> tuple[list[str], dict[str, list[bytes]], dict[tuple[bytes, bytes], set[str]]]:
        pretoken_reprs: dict[str, list[bytes]] = {}
        pretokens: list[str] = []

        for match in re.finditer(train_bpe.PAT, text):
            match_str = match.group()
            pretokens.append(match_str)
            if match_str not in pretoken_reprs:
                match_bytes = tuple(bytes([b]) for b in match_str.encode("UTF-8"))
                pretoken_reprs[match_str] = match_bytes

        pairs_to_pretokens: dict[tuple[bytes, bytes], set[str]] = defaultdict(set)

        for pretoken, pretoken_repr in pretoken_reprs.items():
            for i in range(len(pretoken_repr) - 1):
                pair = (pretoken_repr[i], pretoken_repr[i + 1])
                pairs_to_pretokens[pair].add(pretoken)

        return pretokens, pretoken_reprs, pairs_to_pretokens
