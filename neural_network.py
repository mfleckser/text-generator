from abc import ABC, abstractmethod
import numpy as np

class NeuralNetwork(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def find_gradients(self, x, target):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    @staticmethod
    def sigmoid(x):
        output = np.empty(x.shape)
        for i in range(len(x)):
            val = x[i]
            if val < 0:
                output[i] = np.exp(val) / (1 + np.exp(val))
            else:
                output[i] = 1 / (1 + np.exp(-val)) # easter egg hello future people

        return output

    @staticmethod
    def softmax(x): # do the thing w max in case of big numbers, prolly not needed but its fancy
        return np.exp(x - max(x)) / np.exp(x - max(x)).sum()

    @staticmethod
    def id_from_prob(p):
        return np.argmax(p)

    @staticmethod
    def sparse_categorical_cross_entropy_loss(prediction, target):
        target_probs = np.zeros(prediction.shape)
        target_probs[target] = 1

        #cross_entropy = -np.sum(prediction * np.log2(target_probs + np.exp(-15)))

        #return cross_entropy, target_probs
        return target_probs
