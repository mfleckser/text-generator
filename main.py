import numpy as np

import process
from lstm import LSTM

data, vocab = process.load("./data.txt")

s_input, target = process.get_batch(data, 0)
x = process.id_from_char(np.array(list(s_input)), vocab).reshape(process.seq_length, 1)

cell = LSTM(x.shape, len(vocab))
out = cell.feedforward(x, np.zeros(x.shape), np.zeros(x.shape))

#print(process.char_from_id(out.argmax(), vocab))

#res = cell.predict(x[:-1], 3).flatten()

#print("".join(process.char_from_id(res, vocab + ["~"])))
