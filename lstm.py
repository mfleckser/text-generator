import numpy as np
import os
import math
from process import seq_length, get_batches
from neural_network import NeuralNetwork


class LSTM(NeuralNetwork):
    def __init__(self, input_size, len_vocab):
        self.Wf = np.random.uniform(-1, 1, (input_size[0], input_size[0] * 2))
        self.bf = np.full((input_size[0], 1), 0, dtype=np.float64) # not me anymore L

        self.Wi = np.random.uniform(-1, 1, (input_size[0], input_size[0] * 2))
        self.bi = np.full((input_size[0], 1), 0, dtype=np.float64) # nk

        self.Wc = np.random.uniform(-1, 1, (input_size[0], input_size[0] * 2))
        self.bc = np.full((input_size[0], 1), 0, dtype=np.float64)

        self.Wo = np.random.uniform(-1, 1, (input_size[0], input_size[0] * 2))
        self.bo = np.full((input_size[0], 1), 0, dtype=np.float64)

        self.Wy = np.random.uniform(-1, 1, (len_vocab, input_size[0])) # wy not
        self.by = np.zeros((len_vocab, 1), dtype=np.float64)

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
        #print(np.max(np.dot(self.Wi, concat)), np.min(self.Wi))
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
        #print(pred)
        a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters = caches[0]

        #loss, target_probs = NeuralNetwork.sparse_categorical_cross_entropy_loss(pred, target)
        target_probs = NeuralNetwork.sparse_categorical_cross_entropy_loss(pred, target)

        # derivatives w.r.t loss

        dy_unact = pred - target_probs
        da_next = np.dot(dy_unact.T, parameters["Wy"]).T

        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dc_next = da_next * (1 - np.tanh(c_next) ** 2)
        #print(np.max(parameters["Wf"]))
        dcct = dc_next * it * (1 - cct ** 2)
        dit = dc_next * cct * it * (1 - it)
        #print(np.max(it))
        dft = dc_next * c_prev * ft * (1 - ft)
        dc_prev = dc_next * ft

        dx = parameters["Wf"] * dft + parameters["Wi"] * dit + parameters["Wc"] * dcct + parameters["Wo"] * dot

        dWf = np.dot(dft, np.concatenate((xt, a_prev)).T)
        #print(np.max(dft))
        dWi = np.dot(dit, np.concatenate((xt, a_prev)).T)
        dWc = np.dot(dcct, np.concatenate((xt, a_prev)).T)
        dWo = np.dot(dot, np.concatenate((xt, a_prev)).T)

        dbf = dft
        dbi = dit
        dbc = dcct
        dbo = dot

        dWy = np.dot(dy_unact, a_next.T)
        dby = dy_unact
        # hit the griddy every day and night
        # eckel out here again
        # leaving notes for myself tmr
        # hello future self
        # "Natalie always fucking wins I fucking hate everyone" - Isabel
        # Goofy ahh march madness
        # steak
        # Isabel's a bot
        # "What if I bit rn" - Leon
        # we should eat cock or constantinople

        gradients = {
            "dbf": dbf,
            "dWf": dWf,
            "dWi": dWi,
            "dbi": dbi,
            "dWc": dWc,
            "dbc": dbc,
            "dWo": dWo,
            "dbo": dbo,
            "dWy": dWy,
            "dby": dby,
        }

        return gradients#, loss

    def update_from_gradients(self, gradients, learning_rate):
        for nabla in gradients:
            self.Wf -= learning_rate * nabla["dWf"]
            self.bf -= learning_rate * nabla["dbf"]
            self.Wi -= learning_rate * nabla["dWi"]
            self.bi -= learning_rate * nabla["dbi"]
            self.Wc -= learning_rate * nabla["dWc"]
            self.bc -= learning_rate * nabla["dbc"]
            self.Wo -= learning_rate * nabla["dWo"]
            self.bo -= learning_rate * nabla["dbo"]
            self.Wy -= learning_rate * nabla["dWy"]
            self.by -= learning_rate * nabla["dby"]

    def train(self, data):
        for i in range(100):
            for j in range(len(data) // math.floor(seq_length * 1.3)):
                index = j * seq_length
                gradients = []
                #average_loss = 0
                for batch in get_batches(data[index : index + math.floor(seq_length * 1.3)]):
                    x, target = batch
                    #nabla, loss = self.find_gradients(x, target)
                    nabla = self.find_gradients(x, target)
                    gradients.append(nabla)
                    #os.system("clear")
                    #print(loss)
                    #average_loss += loss
                self.update_from_gradients(gradients, 0.1)
                #print(average_loss / len(gradients))

