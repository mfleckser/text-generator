import numpy as np


class LSTM:
    def __init__(self, input_size):
        self.Wf = np.random.random(input_size)
