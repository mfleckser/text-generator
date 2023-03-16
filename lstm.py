from abc import ABC, abstractmethod
import numpy as np
from process import seq_length


class NeuralNetwork(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def train(self, x, target):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x)) # easter egg hello future people

    @staticmethod
    def softmax(x): # do the thing w max in case of big numbers, prolly not needed but its fancy
        return np.exp(x - max(x)) / np.exp(x - max(x)).sum()

    @staticmethod
    def id_from_prob(p):
        return np.argmax(p)

    @staticmethod
    def sparse_categorical_cross_entropy_loss(prediction, target):
        target_probs = np.zeros(prediction.shape())
        target_probs[target] = 1

        cross_entropy = -np.sum(target_probs * np.log2(prediction))

        return cross_entropy


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

    def predict(self, rx, n): # rx: raw input (unpadded), n: number of predictions
        x = np.concatenate((np.full((seq_length - rx.shape[0], 1), 91), rx))
        y = np.empty(n, dtype=np.int8)

        a = np.zeros(x.shape)
        c = np.zeros(x.shape)

        caches = []

        for i in range(n):
            next_char_prob, a, c, cache = self.feedforward(x, a, c)

            next_char = NeuralNetwork.id_from_prob(next_char_prob)
            y[i] = next_char
            x = np.concatenate((x[1:], np.array([[next_char]])))

            caches.append(cache)

        return y, caches

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

