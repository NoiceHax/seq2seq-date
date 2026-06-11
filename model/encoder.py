import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):

        embedded = self.embedding(x)

        outputs, (hidden, cell) = self.lstm(
            embedded
        )

        return outputs, hidden, cell