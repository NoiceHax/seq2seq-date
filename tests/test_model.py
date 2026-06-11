import torch
from data.dataloader import get_dataloaders
from model.seq2seq import Seq2Seq


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

train_loader, _, _, dataset = get_dataloaders(
    "data/dataset.csv"
)

model = Seq2Seq(
    len(dataset.vocab),
    device
).to(device)

input_batch, target_batch = next(
    iter(train_loader)
)

input_batch = input_batch.to(device)
target_batch = target_batch.to(device)

outputs = model(
    input_batch,
    target_batch
)

print(outputs.shape)
