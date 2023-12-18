import numpy as np
import math

class Layer:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(0)
        k = 1 / math.sqrt(hidden_size)
        self.i_weight = np.random.rand(input_size, hidden_size) * 2 * k - k
        self.h_weight = np.random.rand(hidden_size, hidden_size) * 2 * k - k
        self.h_bias = np.random.rand(1, hidden_size) * 2 * k - k
        self.o_weight = np.random.rand(hidden_size, output_size) * 2 * k - k
        self.o_bias = np.random.rand(1, output_size) * 2 * k - k

    def get_params(self):
        return [self.i_weight, self.h_weight, self.h_bias, self.o_weight, self.o_bias]

    def set_params(self, params):
        self.i_weight, self.h_weight, self.h_bias, self.o_weight, self.o_bias = params
    
    def get_weights(self):
        return {
            "i_weight": self.i_weight.tolist(),
            "h_weight": self.h_weight.tolist(),
            "h_bias": self.h_bias.tolist(),
            "o_weight": self.o_weight.tolist(),
            "o_bias": self.o_bias.tolist()
        }

    def set_weights(self, weights):
        self.i_weight = np.array(weights["i_weight"])
        self.h_weight = np.array(weights["h_weight"])
        self.h_bias = np.array(weights["h_bias"])
        self.o_weight = np.array(weights["o_weight"])
        self.o_bias = np.array(weights["o_bias"])