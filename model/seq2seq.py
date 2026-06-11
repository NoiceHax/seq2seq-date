import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder


class Seq2Seq(nn.Module):

    def __init__(
        self,
        vocab_size,
        device
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size
        )

        self.decoder = Decoder(
            vocab_size
        )

        self.device = device

    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(
            batch_size,
            target_len,
            vocab_size
        ).to(self.device)

        _, hidden, cell = self.encoder(
            source
        )

        decoder_input = target[:, 0]

        for t in range(1, target_len):

            output, hidden, cell = self.decoder(
                decoder_input,
                hidden,
                cell
            )

            outputs[:, t] = output

            best_guess = output.argmax(1)

            use_teacher_force = (
                torch.rand(1).item()
                < teacher_forcing_ratio
            )

            decoder_input = (
                target[:, t]
                if use_teacher_force
                else best_guess
            )

        return outputs