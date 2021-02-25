from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from char_lm.tokenizer import CharacterTokenizer
from argparse import ArgumentParser


class CityLanguageModel(pl.LightningModule):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, ignore_index: int):
        super().__init__()

        self.save_hyperparameters()

        self.embedding = nn.Embedding.from_pretrained(torch.eye(vocab_size))

        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn, _ = self.lstm(embedded)

        output = self.fc(rnn) # output: batch_size x seq len x vocab size

        # need to change output shape to: batch_size x vocab size x seq len
        output = output.permute(0, 2, 1)

        return output

    def training_step(self, batch, batch_idx):
        # batch: batch size x seq len (variable)
        x = batch[:, :-1]
        y = batch[:, 1:]

        output = self(x)
        loss = self.criterion(output, y)

        return loss

    def validation_step(self, batch, batch_idx):
        # batch: batch size x seq len (variable)
        x = batch[:, :-1]
        y = batch[:, 1:]

        output = self.forward(x)
        loss = self.criterion(output, y)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters())


if __name__ == "__main__":
    tokenizer = CharacterTokenizer.from_file(Path("models/tokenizer.dill"))

    def collate_batch(batch):
        return tokenizer.encode_batch(batch)

    dataset = load_dataset(
        "text",
        data_files={
            "train": "data/dataset/city/train.txt",
            "val": "data/dataset/city/val.txt",
            "test": "data/dataset/city/test.txt",
        },
    )

    train_loader = DataLoader(dataset["train"]["text"], batch_size=32, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(dataset["val"]["text"], batch_size=32, shuffle=False, collate_fn=collate_batch)

    lm = CityLanguageModel(tokenizer.get_vocab_size(), hidden_size=64, num_layers=2, ignore_index=tokenizer.pad_index)

    trainer = pl.Trainer(gpus=[0], gradient_clip_val=1.0, max_epochs=100)

    trainer.fit(model=lm, train_dataloader=train_loader, val_dataloaders=val_loader)
