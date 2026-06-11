from data.dataloader import get_dataloaders

train_loader, val_loader, test_loader, dataset = get_dataloaders(
    "data/dataset.csv"
)

print("Vocabulary size:", len(dataset.vocab))
print("Max input length:", dataset.max_input_len)
print("Max target length:", dataset.max_target_len)

for batch in train_loader:

    input_batch, target_batch = batch

    print("Input batch shape:")
    print(input_batch.shape)

    print("Target batch shape:")
    print(target_batch.shape)

    break
