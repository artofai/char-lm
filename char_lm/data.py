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

tokenizer = CharacterTokenizer.from_file(Path("models/tokenizer.dill"))


class CityDataModule(LightningDataModule):
    def __init__():
        super().__init__()


def tokenize(batch):
    return {"encoded": tokenizer.encode_batch(batch["text"])}


dset.set_transform(transform=tokenize)


dm = LightningDataModule.from_datasets(dset["train"], val_dataset=dset["val"], batch_size=16)


print(dset)

print(dset["train"][:7])
