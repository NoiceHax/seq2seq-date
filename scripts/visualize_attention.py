import torch
import matplotlib.pyplot as plt
import numpy as np

from model.attn_seq2seq import AttnSeq2Seq
from data.dataset import DateDataset


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset = DateDataset(
    "data/dataset.csv"
)

model = AttnSeq2Seq(
    len(dataset.vocab),
    device
).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/attn_seq2seq_model.pth",
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


def predict_with_attention(text):

    source = encode_input(text)

    if source is None:
        return None, None, None

    with torch.no_grad():

        encoder_outputs, hidden, cell = (
            model.encoder(source)
        )

        decoder_input = torch.tensor(
            [dataset.char2idx["<SOS>"]]
        ).to(device)

        prediction = []
        all_weights = []

        for _ in range(
            dataset.max_target_len
        ):

            output, hidden, cell, weights = (
                model.decoder(
                    decoder_input,
                    hidden,
                    cell,
                    encoder_outputs
                )
            )

            all_weights.append(
                weights.squeeze(0).cpu().numpy()
            )

            best_guess = output.argmax(1)

            token = best_guess.item()

            if (
                token
                == dataset.char2idx["<EOS>"]
            ):
                break

            prediction.append(
                dataset.idx2char[token]
            )

            decoder_input = best_guess

    pred_str = "".join(prediction)
    attn_matrix = np.array(all_weights)

    # Build input character labels
    input_chars = ["<SOS>"]
    input_chars += list(text)
    input_chars += ["<EOS>"]

    pad_count = (
        dataset.max_input_len
        - len(input_chars)
    )

    input_chars += ["<PAD>"] * pad_count

    return pred_str, attn_matrix, input_chars


def plot_attention(
    input_text,
    pred_str,
    attn_matrix,
    input_chars,
    save_path=None
):

    output_chars = list(pred_str)

    # Trim to actual input length (no PAD)
    actual_len = len(input_text) + 2
    trimmed_attn = attn_matrix[
        :, :actual_len
    ]
    trimmed_chars = input_chars[:actual_len]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    im = ax.imshow(
        trimmed_attn,
        cmap="YlOrRd",
        aspect="auto"
    )

    ax.set_xticks(range(len(trimmed_chars)))
    ax.set_xticklabels(
        trimmed_chars,
        rotation=45,
        ha="right",
        fontsize=11
    )

    ax.set_yticks(range(len(output_chars)))
    ax.set_yticklabels(
        output_chars,
        fontsize=11
    )

    ax.set_xlabel(
        "Input Characters",
        fontsize=13
    )
    ax.set_ylabel(
        "Output Characters",
        fontsize=13
    )

    ax.set_title(
        f'Attention: "{input_text}" → '
        f'"{pred_str}"',
        fontsize=14,
        fontweight="bold"
    )

    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )
        print(f"Saved: {save_path}")

    plt.show()


if __name__ == "__main__":

    test_inputs = [
        "march 5, 3000",
        "December 31 2099",
        "February 29, 2004",
        "07/04/1776",
        "25 JANUARY, 2050",
    ]

    for i, text in enumerate(test_inputs):

        pred, attn, chars = (
            predict_with_attention(text)
        )

        if pred is not None:

            print(
                f'"{text}" → "{pred}"'
            )

            plot_attention(
                text,
                pred,
                attn,
                chars,
                save_path=(
                    f"outputs/attention_heatmap_{i+1}.png"
                )
            )
