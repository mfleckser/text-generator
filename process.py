import numpy as np

seq_length = 96


def get_vocab(data):
    return sorted(set(data))


def load(path):
    file = open(path, "r", encoding="utf-8")
    data = file.read()
    file.close()

    return data


def rid_from_char(c, vocab):
    return vocab.index(c)

def rchar_from_id(id, vocab):
    return vocab[id]

id_from_char = np.vectorize(rid_from_char, excluded=[1])
char_from_id = np.vectorize(rchar_from_id, excluded=[1])


def get_batch(data, index):
    vocab = get_vocab(data)

    x = np.array(list(data[index : index + seq_length]))
    target = data[index + seq_length]

    return id_from_char(x, vocab).reshape(seq_length, 1), id_from_char(target, vocab)
