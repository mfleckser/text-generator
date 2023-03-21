import numpy as np
from process import seq_length
from neural_network import NeuralNetwork


class LSTM(NeuralNetwork):
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

        self.len_vocab = len_vocab

    @property
    def parameters(self):
        return {
            "Wf": self.Wf,
            "bf": self.bf,
            "Wi": self.Wi,
            "bi": self.bi,
            "Wc": self.Wc,
            "bc": self.bc,
            "Wo": self.Wo,
            "bo": self.bo,
            "Wy": self.Wy,
            "by": self.by,
        }

    def feedforward(self, rxt, a_prev, c_prev): # rxt: raw input at time t
        xt = rxt / max(rxt) # normalize input so dot product doesn't go wild
        concat = np.concatenate((xt, a_prev))

        ft = NeuralNetwork.sigmoid(np.dot(self.Wf, concat) + self.bf)
        it = NeuralNetwork.sigmoid(np.dot(self.Wi, concat) + self.bi)
        cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
        c_next = ft * c_prev + it * cct # leaving a random comment because I want to
        
        ot = NeuralNetwork.sigmoid(np.dot(self.Wo, concat) + self.bo)
        a_next = ot * np.tanh(c_next)

        yt = NeuralNetwork.softmax(np.dot(self.Wy, a_next) + self.by)
        
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, self.parameters)

        return yt, a_next, c_next, cache

    def predict(self, rx, n, probs=False): # rx: raw input (unpadded), n: number of predictions
        x = np.concatenate((np.full((seq_length - rx.shape[0], 1), 91), rx))
        if probs:
            out_shape = (n, self.len_vocab, 1)
            out_type = np.float32
        else:
            out_shape = n
            out_type = np.int8
        y = np.empty(out_shape, dtype=out_type)

        a = np.zeros(x.shape)
        c = np.zeros(x.shape)

        caches = []

        for i in range(n):
            next_char_prob, a, c, cache = self.feedforward(x, a, c)

            next_char = NeuralNetwork.id_from_prob(next_char_prob)

            if probs:
                y[i] = next_char_prob
            else:
                y[i] = next_char

            x = np.concatenate((x[1:], np.array([[next_char]])))

            caches.append(cache)

        return y, caches

    def find_gradients(self, x, target):
        prediction, caches = self.predict(x, 1, probs=True)
        pred = prediction[0]
        a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters = caches[0]

        loss = NeuralNetwork.sparse_categorical_cross_entropy_loss(pred, target)

        # derivatives w.r.t loss

        dy_unact = pred - target
        da_next = dy_unact * parameters["Wy"]

        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dc_next = da_next * (1 - np.tanh(c_next) ** 2)
        dcct = dc_next * it * (1 - cct ** 2)
        dit = dc_next * cct * it * (1 - it)
        dft = dc_next * c_prev * ft (1 - ft)
        dc_prev = dc_next * ft


        print(loss)

    def train(self, x, target):
        pass


class Encoder(NeuralNetwork):
    def __init__(self, output_size):
        self.weights = np.random.random((1, output_size))
        self.biases = np.zeros((output_size, 1))

    def encode(self, x):
        return x * self.weights + self.biases

    def train(self, x, target):
        pass
