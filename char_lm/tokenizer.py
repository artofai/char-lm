import logging
from collections import Counter, defaultdict
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    def __init__(self) -> None:
        self.sos_token = "^"
        self.eos_token = "$"
        self.oov_token = "@"
        self.pad_token = "#"
        self.pad_index = 0

    def train(self, corpus_file: Path):
        with corpus_file.open("rt") as f:
            text = f.read().replace("\n", "")

        char_freq = Counter(text)
        logging.info(f"Most frequent tokens {char_freq.most_common(10)}")
        self.idx2tok = list([self.pad_token, self.sos_token, self.eos_token, self.oov_token] + list(char_freq.keys()))
        self.tok2idx = {tok: idx for idx, tok in enumerate(self.idx2tok)}
        self.tok2idx = defaultdict(lambda: self.tok2idx[self.oov_token], self.tok2idx)
        logger.info(f"Created vocabulary with {self.get_vocab_size()} tokens")

    def get_vocab_size(self) -> int:
        return len(self.tok2idx)

    def encode(self, s: str, include_special_tokens=True) -> torch.LongTensor:
        if include_special_tokens:
            s = [self.sos_token] + list(s) + [self.eos_token]

        indices = [self.tok2idx[c] for c in s]
        return torch.LongTensor(indices)

    def decode(self, t: torch.LongTensor) -> str:
        decoded = [self.idx2tok[i] for i in t]
        return "".join(decoded)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ct = CharacterTokenizer()
    ct.train(Path("data/dataset/city/full.txt"))

    print("Vocab size:", ct.get_vocab_size())

    sample_sequence = "Abbeville"
    print("Sequence", sample_sequence)
    encoded = ct.encode(sample_sequence)
    print("Encoded", encoded)
    decoded = ct.decode(encoded)
    print("Decoded", decoded)

    print("Vocabulary:")
    print(ct.idx2tok)