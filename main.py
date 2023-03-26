import numpy as np

import process
from lstm import LSTM

data = process.load("./data.txt")
vocab = process.get_vocab(data)

x, target = process.get_batch(data, 0)

cell = LSTM(x.shape, len(vocab))
#out = cell.feedforward(x, np.zeros(x.shape), np.zeros(x.shape))

#print(process.char_from_id(out.argmax(), vocab))

res = cell.predict(x[:-1], 6)[0].flatten()

print("".join(process.char_from_id(res, vocab + ["~"])))

#print("".join(process.char_from_id(cell.predict(x, 20)[0], vocab + ["~"])))

cell.train(data)


res = cell.predict(x[:-1], 15)[0].flatten()
print("".join(process.char_from_id(x[:-1].T[0], vocab + ["~"])), end="")
print("".join(process.char_from_id(res, vocab + ["~"])), end="ENDDDD")