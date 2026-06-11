import torch
from tqdm import tqdm

from data.dataloader import get_dataloaders
from model.seq2seq import Seq2Seq


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def decode_sequence(indices, dataset):

    chars = []

    for idx in indices:

        token = dataset.idx2char[idx.item()]

        if token == "<EOS>":
            break

        if token not in ("<SOS>", "<PAD>"):
            chars.append(token)

    return "".join(chars)


def evaluate_model(
    model,
    test_loader,
    dataset,
    model_name="Model",
    show_errors=True,
    max_errors=20
):

    model.eval()

    correct = 0
    total = 0
    errors = []

    with torch.no_grad():

        for source, target in tqdm(
            test_loader,
            desc=f"Evaluating {model_name}"
        ):

            source = source.to(device)
            target = target.to(device)

            batch_size = source.shape[0]
            target_len = target.shape[1]
            vocab_size = model.decoder.fc.out_features

            outputs = torch.zeros(
                batch_size,
                target_len,
                vocab_size
            ).to(device)

            encoder_out = model.encoder(
                source
            )

            if len(encoder_out) == 3:
                encoder_outputs, hidden, cell = encoder_out
            else:
                encoder_outputs, (hidden, cell) = None, encoder_out

            decoder_input = target[:, 0]

            for t in range(1, target_len):

                if type(model.decoder).__name__ == "AttnDecoder":
                    output, hidden, cell, _ = (
                        model.decoder(
                            decoder_input,
                            hidden,
                            cell,
                            encoder_outputs
                        )
                    )
                else:
                    output, hidden, cell = (
                        model.decoder(
                            decoder_input,
                            hidden,
                            cell
                        )
                    )

                outputs[:, t] = output

                decoder_input = output.argmax(1)

            predictions = outputs.argmax(2)

            for i in range(batch_size):

                pred_str = decode_sequence(
                    predictions[i],
                    dataset
                )

                target_str = decode_sequence(
                    target[i],
                    dataset
                )

                total += 1

                if pred_str == target_str:
                    correct += 1
                else:
                    input_str = decode_sequence(
                        source[i],
                        dataset
                    )
                    errors.append(
                        (input_str, target_str, pred_str)
                    )

    accuracy = correct / total * 100

    print(f"\n{model_name} Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Exact Match Accuracy: {accuracy:.2f}%")

    if show_errors and errors:

        print(f"\n  Sample Errors "
              f"(showing {min(max_errors, len(errors))}"
              f"/{len(errors)}):")

        for inp, tgt, pred in errors[:max_errors]:

            print(
                f"    {inp} -> {pred} "
                f"(expected {tgt})"
            )

    return accuracy


if __name__ == "__main__":

    _, _, test_loader, dataset = (
        get_dataloaders("data/dataset.csv")
    )

    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Vocab size: {len(dataset.vocab)}")

    # --- Vanilla Seq2Seq ---

    vanilla_model = Seq2Seq(
        len(dataset.vocab),
        device
    ).to(device)

    vanilla_model.load_state_dict(
        torch.load(
            "checkpoints/seq2seq_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    vanilla_acc = evaluate_model(
        vanilla_model,
        test_loader,
        dataset,
        model_name="Vanilla Seq2Seq"
    )

    # --- Attention Seq2Seq ---

    try:

        from model.attn_seq2seq import AttnSeq2Seq

        attn_model = AttnSeq2Seq(
            len(dataset.vocab),
            device
        ).to(device)

        attn_model.load_state_dict(
            torch.load(
                "checkpoints/attn_seq2seq_model.pth",
                map_location=device,
                weights_only=True
            )
        )

        attn_acc = evaluate_model(
            attn_model,
            test_loader,
            dataset,
            model_name="Attention Seq2Seq"
        )

        print("\n" + "=" * 50)
        print("COMPARISON")
        print("=" * 50)
        print(
            f"{'Model':<25} | "
            f"{'Exact Match':>12}"
        )
        print("-" * 42)
        print(
            f"{'Vanilla Seq2Seq':<25} | "
            f"{vanilla_acc:>11.2f}%"
        )
        print(
            f"{'Attention Seq2Seq':<25} | "
            f"{attn_acc:>11.2f}%"
        )
        print("=" * 50)

    except (FileNotFoundError, ImportError):

        print(
            "\nAttention model not yet trained. "
            "Run train_attention.py first."
        )
