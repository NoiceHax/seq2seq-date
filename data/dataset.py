import torch
from torch.utils.data import Dataset
import pandas as pd


class DateDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.inputs = self.df["input"].tolist()
        self.targets = self.df["target"].tolist()

        all_text = "".join(self.inputs + self.targets)

        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>"]

        chars = sorted(list(set(all_text)))

        self.vocab = self.special_tokens + chars

        self.char2idx = {
            ch: idx for idx, ch in enumerate(self.vocab)
        }

        self.idx2char = {
            idx: ch for ch, idx in self.char2idx.items()
        }

        self.max_input_len = max(
            len(x) for x in self.inputs
        ) + 2

        self.max_target_len = max(
            len(x) for x in self.targets
        ) + 2

    def encode(self, text, max_len):

        sequence = [
            self.char2idx["<SOS>"]
        ]

        sequence += [
            self.char2idx[ch]
            for ch in text
        ]

        sequence.append(
            self.char2idx["<EOS>"]
        )

        while len(sequence) < max_len:
            sequence.append(
                self.char2idx["<PAD>"]
            )

        return torch.tensor(sequence)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        input_seq = self.encode(
            self.inputs[idx],
            self.max_input_len
        )

        target_seq = self.encode(
            self.targets[idx],
            self.max_target_len
        )

        return input_seq, target_seq
