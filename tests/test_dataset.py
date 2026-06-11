from data.dataset import DateDataset

dataset = DateDataset("data/dataset.csv")

print("Dataset size:", len(dataset))
print("Vocabulary size:", len(dataset.vocab))

x, y = dataset[0]

print("Encoded Input:")
print(x)

print("Encoded Target:")
print(y)

print("\nVocabulary:")
print(dataset.vocab)
