from pathlib import Path

import pytest
import torch

from char_lm.tokenizer import CharacterTokenizer


@pytest.fixture
def tokenizer():
    corpus_path = Path("tests/fixtures/tiny_corpus.txt")
    tokenizer = CharacterTokenizer()
    tokenizer.train(corpus_path)
    return tokenizer


def test_vocab_size(tokenizer: CharacterTokenizer):

    # 6 letters + 4 special tokens
    assert tokenizer.get_vocab_size() == 6 + 4


def test_encode_decode_without_special_chars(tokenizer: CharacterTokenizer):
    text = "abc"
    encoded = tokenizer.encode(text, include_special_tokens=False)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_encode_decode_with_special_chars(tokenizer: CharacterTokenizer):
    text = "abc"
    encoded = tokenizer.encode(text, include_special_tokens=True)
    decoded = tokenizer.decode(encoded)

    assert decoded == tokenizer.sos_token + text + tokenizer.eos_token


def test_encode_with_oov(tokenizer: CharacterTokenizer):
    text = "abcX"
    encoded = tokenizer.encode(text, include_special_tokens=False)
    assert encoded[3] == tokenizer.tok2idx[tokenizer.oov_token]


def test_encode_decode_with_oov(tokenizer: CharacterTokenizer):
    text = "abcX"
    encoded = tokenizer.encode(text, include_special_tokens=True)
    decoded = tokenizer.decode(encoded)
    assert decoded == tokenizer.sos_token + "abc" + tokenizer.oov_token + tokenizer.eos_token


def test_encode_batch_with_var_size(tokenizer: CharacterTokenizer):
    texts = ["a", "ab", "abc"]
    batch = tokenizer.encode_batch(texts)
    # batch x seq len; 3x5
    assert batch.shape == (3, 5)
    # 2 pads in 0th example
    assert torch.equal(batch[0, 3:], torch.LongTensor([0, 0]))
    # 0 pads in last example
    assert torch.all(batch[2, :] != torch.LongTensor([0, 0, 0, 0, 0]))


def test_serialize_deserialize(tokenizer: CharacterTokenizer, tmp_path):
    serialization_path = tmp_path / "tokenizer.pickle"
    texts = ["a", "b", "abc"]
    encoded = tokenizer.encode_batch(texts)

    tokenizer.to_file(serialization_path)

    loaded_tok = CharacterTokenizer.from_file(serialization_path)
    loaded_encoded = loaded_tok.encode_batch(texts)

    assert torch.all(encoded == loaded_encoded)
