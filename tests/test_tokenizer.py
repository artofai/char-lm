from pathlib import Path

import pytest

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
