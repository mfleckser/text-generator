import process

data, vocab = process.load("./data.txt")

batches = process.get_batch(data)
print(next(batches))
print(next(batches))
