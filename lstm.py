import numpy as np
from process import seq_length


class LSTM:
    def __init__(self, input_size, len_vocab):
        self.Wf = np.random.random((input_size[0], input_size[0] * 2))
        self.bf = np.full((input_size[0], 1), -30) # not me anymore L

        self.Wi = np.random.random((input_size[0], input_size[0] * 2))
        self.bi = np.full((input_size[0], 1), -30) # nk

        self.Wc = np.random.random((input_size[0], input_size[0] * 2))
        self.bc = np.full((input_size[0], 1), -30)

        self.Wo = np.random.random((input_size[0], input_size[0] * 2))
        self.bo = np.full((input_size[0], 1), -30)

        self.Wy = np.random.random((len_vocab, input_size[0])) # wy not
        self.by = np.zeros((len_vocab, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x)) # easter egg hello future people

    @staticmethod
    def softmax(x): # do the thing w max in case of big numbers, prolly not needed but its fancy
        return np.exp(x - max(x)) / np.exp(x - max(x)).sum()

    @staticmethod
    def id_from_prob(p):
        return np.argmax(p)

    def feedforward(self, rxt, a_prev, c_prev): # rxt: raw input at time t
        xt = rxt / max(rxt) # normalize input so dot product doesn't go wild
        concat = np.concatenate((xt, a_prev))

        ft = LSTM.sigmoid(np.dot(self.Wf, concat) + self.bf)
        it = LSTM.sigmoid(np.dot(self.Wi, concat) + self.bi)
        cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
        c_next = ft * c_prev + it * cct # leaving a random comment because I want to
        
        ot = LSTM.sigmoid(np.dot(self.Wo, concat) + self.bo)
        a_next = ot * np.tanh(c_next)

        yt = LSTM.softmax(np.dot(self.Wy, a_next) + self.by)
        return yt

    def predict(self, rx, n): # rx: raw input (unpadded), n: number of predictions
        x = np.concatenate((np.full((seq_length - rx.shape[0], 1), 91), rx))
        y = np.empty(n, dtype=np.int8)

        for i in range(n):
            next_char = LSTM.id_from_prob(self.feedforward(x, np.zeros(x.shape), np.zeros(x.shape)))
            y[i] = next_char
            x = np.concatenate((x[1:], np.array([[next_char]])))

        return y

