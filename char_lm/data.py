from pathlib import Path

import datasets
from pytorch_lightning import LightningDataModule

from char_lm.tokenizer import CharacterTokenizer

dset = datasets.load_dataset(
    "text",
    data_files={
        "train": "data/dataset/city/train.txt",
        "val": "data/dataset/city/val.txt",
        "test": "data/dataset/city/test.txt",
    },
)

tokenizer = CharacterTokenizer.from_file(Path("model/tokenizer.dill"))


def tokenize(batch):
    return {"encoded": tokenizer.encode_batch(batch["text"])}


dset.set_transform(transform=tokenize)

print(dset)

print(dset["train"][:7])
