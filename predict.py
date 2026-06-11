import torch
from model.seq2seq import Seq2Seq
from data.dataset import DateDataset


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset = DateDataset(
    "data/dataset.csv"
)

model = Seq2Seq(
    len(dataset.vocab),
    device
).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/seq2seq_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


def encode_input(text):

    sequence = [
        dataset.char2idx["<SOS>"]
    ]

    for ch in text:

        if ch in dataset.char2idx:
            sequence.append(
                dataset.char2idx[ch]
            )
        else:
            print(
                f"Unknown character: {ch}"
            )
            return None

    sequence.append(
        dataset.char2idx["<EOS>"]
    )

    while len(sequence) < dataset.max_input_len:
        sequence.append(
            dataset.char2idx["<PAD>"]
        )

    return torch.tensor(
        sequence
    ).unsqueeze(0).to(device)


def predict(text):

    source = encode_input(text)

    if source is None:
        return "Invalid input"

    with torch.no_grad():

        _, hidden, cell = model.encoder(
            source
        )

        decoder_input = torch.tensor(
            [dataset.char2idx["<SOS>"]]
        ).to(device)

        prediction = []

        for _ in range(
            dataset.max_target_len
        ):

            output, hidden, cell = (
                model.decoder(
                    decoder_input,
                    hidden,
                    cell
                )
            )

            best_guess = (
                output.argmax(1)
            )

            token = (
                best_guess.item()
            )

            if (
                token
                == dataset.char2idx["<EOS>"]
            ):
                break

            prediction.append(
                dataset.idx2char[token]
            )

            decoder_input = (
                best_guess
            )

    return "".join(prediction)


while True:

    text = input(
        "\nEnter date: "
    )

    print(
        "Prediction:",
        predict(text)
    )