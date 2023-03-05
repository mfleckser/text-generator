import numpy as np



def load(path):
    file = open(path, "r")
    data = file.read()

    vocab = sorted(set(data))

    # id_from_char = np.vectorize(lambda c: vocab.index(c))
    # char_from_id = np.vectorize(lambda id: vocab[id])

    file.close()

    return data, vocab


def rid_from_char(c, vocab):
    return vocab.index(c)

def rchar_from_id(id, vocab):
    return vocab[id]

id_from_char = np.vectorize(rid_from_char)
char_from_id = np.vectorize(rchar_from_id)


def get_batch(data, index):
    seq_length = 96 # num of chars in each input, chosen to minimize leftover at end of data

    return data[index : index + seq_length], data[index + seq_length + 1]


