"""BPE Tokenizer — fast-gpt-lab
Byte-Pair Encoding from first principles.
Reference: Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units"
"""
import re
import json
import collections
from pathlib import Path
from typing import Optional


# ─── Regex pattern (same as GPT-2) ───────────────────────────────────────────
# Splits input into chunks: contractions, letters, numbers, spaces, others
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer — the backbone of GPT-2..Llama tokenisation.
    
    Algorithm:
    1. Start with byte-level vocabulary {0..255}
    2. Count all adjacent byte-pair frequencies in the corpus
    3. Merge the most frequent pair → new token
    4. Repeat num_merges times
    
    This is O(N·V) where N = corpus size, V = vocab size.
    """

    def __init__(self):
        # byte → single-char string (for readable repr)
        self._byte_encoder = self._build_byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self.vocab: dict[str, int] = {}
        self.merges: dict[tuple[str, str], int] = {}
        self._cache: dict[str, tuple[str, ...]] = {}
        self.special_tokens: dict[str, int] = {}
        self.pat = GPT2_PATTERN

    # ── Vocabulary building ───────────────────────────────────────────────────

    def train(self, corpus: str, vocab_size: int = 512, verbose: bool = True) -> None:
        """Train BPE on a raw text corpus."""
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte alphabet)"
        num_merges = vocab_size - 256 - len(self.special_tokens)

        # Initialise byte-level vocabulary
        self.vocab = {ch: i for i, ch in enumerate(self._byte_encoder.values())}
        self.merges = {}
        word_freqs = self._build_word_frequencies(corpus)

        if verbose:
            print(f"🔤 Training BPE | target vocab={vocab_size} | merges={num_merges}")

        for merge_idx in range(num_merges):
            pair_freqs = self._count_pairs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged = "".join(best_pair)

            # Record the merge
            self.merges[best_pair] = merge_idx
            self.vocab[merged] = 256 + merge_idx

            # Apply merge to all words
            word_freqs = self._merge_pair(best_pair, word_freqs)

            if verbose and merge_idx % 100 == 0:
                print(f"  [{merge_idx:4d}/{num_merges}] merge: {best_pair!r} → {merged!r} "
                      f"(freq={pair_freqs[best_pair]:,})")

        self._cache.clear()
        if verbose:
            print(f"✅ BPE training done. Vocab size: {len(self.vocab)}")

    def _build_word_frequencies(self, corpus: str) -> dict[str, int]:
        """Segment corpus via GPT-2 regex then byte-encode each chunk."""
        freqs: dict[str, int] = collections.Counter()
        for token in re.findall(self.pat, corpus):
            word = " ".join(self._byte_encoder[b] for b in token.encode("utf-8"))
            freqs[word] += 1
        return freqs

    @staticmethod
    def _count_pairs(word_freqs: dict[str, int]) -> dict[tuple, int]:
        pairs: dict[tuple, int] = collections.Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _merge_pair(pair: tuple[str, str], word_freqs: dict[str, int]) -> dict[str, int]:
        merged_token = "".join(pair)
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_freqs: dict[str, int] = {}
        for word, freq in word_freqs.items():
            new_word = pattern.sub(merged_token, word)
            new_freqs[new_word] = freq
        return new_freqs

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        tokens = []
        for chunk in re.findall(self.pat, text):
            byte_str = " ".join(self._byte_encoder[b] for b in chunk.encode("utf-8"))
            bpe_tokens = self._bpe(byte_str)
            tokens.extend(self.vocab[t] for t in bpe_tokens)
        return tokens

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        text = "".join(id_to_token[i] for i in ids)
        # text is still byte-repr; decode back to unicode
        return bytearray([self._byte_decoder[c] for c in text]).decode("utf-8", errors="replace")

    def _bpe(self, token: str) -> tuple[str, ...]:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token.split())
        pairs = list(zip(word, word[1:]))
        if not pairs:
            return word
        while True:
            # Find the highest-priority merge available in the current word
            bigram = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if bigram not in self.merges:
                break
            a, b = bigram
            merged = a + b
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(a, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            pairs = list(zip(word, word[1:]))
        self._cache[token] = word
        return word

    # ── Special tokens ────────────────────────────────────────────────────────

    def add_special_token(self, token: str) -> int:
        idx = len(self.vocab)
        self.vocab[token] = idx
        self.special_tokens[token] = idx
        return idx

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(path / "merges.json", "w", encoding="utf-8") as f:
            serializable = [list(k) + [v] for k, v in self.merges.items()]
            json.dump(serializable, f)
        print(f"💾 Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        path = Path(path)
        with open(path / "vocab.json", encoding="utf-8") as f:
            tok.vocab = json.load(f)
        with open(path / "merges.json") as f:
            raw = json.load(f)
            tok.merges = {(r[0], r[1]): r[2] for r in raw}
        return tok

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_byte_encoder() -> dict[int, str]:
        """
        GPT-2 byte encoder: maps every byte 0-255 to a printable unicode char.
        Avoids issues with whitespace/control chars in vocab files.
        """
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, (chr(c) for c in cs)))

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={len(self)}, merges={len(self.merges)})"
