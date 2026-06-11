import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size=256):
        super().__init__()

        self.W_encoder = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.W_decoder = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.V = nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(
        self,
        decoder_hidden,
        encoder_outputs
    ):

        # decoder_hidden: (1, batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)

        hidden = decoder_hidden.squeeze(0)
        hidden = hidden.unsqueeze(1)

        # (batch, seq_len, hidden)
        energy = torch.tanh(
            self.W_encoder(encoder_outputs)
            + self.W_decoder(hidden)
        )

        # (batch, seq_len, 1) → (batch, seq_len)
        scores = self.V(energy).squeeze(2)

        weights = torch.softmax(scores, dim=1)

        # (batch, 1, seq_len) @ (batch, seq_len, hidden)
        # → (batch, 1, hidden) → (batch, hidden)
        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, weights
