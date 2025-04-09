from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics import train_bpe


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Constructs a tokenizer from a vocab, list of merges, and (optionally) list of special tokens.
        """
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}

        # Sort special tokens by length in descending order to prioritize longer matches
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

        # Add special tokens to vocab and vocab_inv with unique token ids
        if special_tokens:
            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Constructs a Tokenizer from a serialized vocab, list of merges, and (optionally) list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encodes an input text into a sequence of token IDs, handling special tokens."""
        if not self.special_tokens:
            return self._encode_chunk(text)

        # If we have special tokens, split on them, keeping delimiters
        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                # this is a special token
                ids.append(self.vocab_inv[part.encode("UTF-8")])
            else:
                # this is ordinary text
                ids.extend(self._encode_chunk(part))
        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        """Encodes an input text chunk into a sequence of token IDs."""
        pretokens, pretoken_reprs = self._pretokenize(text)

        # Merge each pretoken using the BPE rules, in ascending rank order
        for p in pretokens:
            pretoken_reprs[p] = self._merge_subword(pretoken_reprs[p])

        # Convert each final subword to its ID
        ids = []
        for p in pretokens:
            for subword in pretoken_reprs[p]:
                ids.append(self.vocab_inv[subword])

        return ids

    def _merge_subword(self, rep: list[bytes]) -> list[bytes]:
        """
        Given a list of subword units (bytes), repeatedly merges adjacent pairs
        in ascending rank order until no more merges are found.
        """
        while True:
            best_rank = float("inf")
            best_idx = None

            # Scan adjacent pairs
            for i in range(len(rep) - 1):
                pair = (rep[i], rep[i + 1])
                rank = self.merges_dict.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            # If no merges found, we're done
            if best_idx is None:
                return rep

            # Merge the best pair
            merged = rep[best_idx] + rep[best_idx + 1]  # Concatenate bytes
            rep = rep[:best_idx] + [merged] + rep[best_idx + 2 :]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yields token IDs lazile from an iterable of strings (e.g., a file handle)."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decodes a sequence of token IDs into text."""
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")

    def _pretokenize(self, text: str) -> tuple[list[str], dict[str, list[bytes]]]:
        """Splits text into 'pretokens' and builds an initial byte representation for each."""
        pretokens: list[str] = []
        pretoken_reprs: dict[str, list[bytes]] = {}

        for match in re.finditer(train_bpe.PAT, text):
            match_str = match.group()
            pretokens.append(match_str)

            # Each character → single bytes: e.g. "abc" -> [b'a', b'b', b'c']
            if match_str not in pretoken_reprs:
                match_bytes = list(bytes([b]) for b in match_str.encode("UTF-8"))
                pretoken_reprs[match_str] = match_bytes

        return pretokens, pretoken_reprs
