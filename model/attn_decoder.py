import torch
import torch.nn as nn
from model.attention import BahdanauAttention


class AttnDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=1
    ):
        super().__init__()

        self.attention = BahdanauAttention(
            hidden_size
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(
        self,
        x,
        hidden,
        cell,
        encoder_outputs
    ):

        x = x.unsqueeze(1)

        embedded = self.embedding(x)

        context, weights = self.attention(
            hidden,
            encoder_outputs
        )

        context = context.unsqueeze(1)

        lstm_input = torch.cat(
            [embedded, context],
            dim=2
        )

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        prediction = self.fc(
            output.squeeze(1)
        )

        return prediction, hidden, cell, weights
