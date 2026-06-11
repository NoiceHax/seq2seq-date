import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.dataloader import get_dataloaders
from model.attn_seq2seq import AttnSeq2Seq


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)

train_loader, val_loader, _, dataset = (
    get_dataloaders(
        "data/dataset.csv"
    )
)

model = AttnSeq2Seq(
    len(dataset.vocab),
    device
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(
    ignore_index=dataset.char2idx["<PAD>"]
)

epochs = 15


for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs}"
    )

    for source, target in progress_bar:

        source = source.to(device)
        target = target.to(device)

        outputs, _ = model(
            source,
            target
        )

        outputs = outputs[:, 1:].reshape(
            -1,
            outputs.shape[2]
        )

        target = target[:, 1:].reshape(
            -1
        )

        loss = criterion(
            outputs,
            target
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item()
        )

    avg_loss = (
        total_loss
        / len(train_loader)
    )

    print(
        f"Epoch {epoch+1} Loss: "
        f"{avg_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "checkpoints/attn_seq2seq_model.pth"
)

print("Attention model saved!")
