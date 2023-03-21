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
        target_probs = np.zeros(prediction.shape)
        target_probs[target] = 1

        cross_entropy = -np.sum(target_probs * np.log2(prediction))

        return cross_entropy
