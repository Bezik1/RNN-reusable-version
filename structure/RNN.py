from helpers.functions import mse, mse_grad
from structure.Layer import Layer
import numpy as np
import json

class RNN:
    def __init__(self, hyperparameters):
        self.learning_rate = hyperparameters.learning_rate
        self.epochs = hyperparameters.epochs
        self.input_size = hyperparameters.input_size
        self.hidden_size = hyperparameters.hidden_size
        self.output_size = hyperparameters.output_size
        
        self.layers = self.init_layers()

    def init_layers(self):
        return [Layer(self.input_size, self.hidden_size, self.output_size)]

    def forward(self, x):
        hiddens = []
        outputs = []
        for i in range(len(self.layers)):
            i_weight, h_weight, h_bias, o_weight, o_bias = self.layers[i].get_params()
            hidden = np.zeros((x.shape[0], i_weight.shape[1]))
            output = np.zeros((x.shape[0], o_weight.shape[1]))
            for j in range(x.shape[0]):
                input_x = x[j, :][np.newaxis, :] @ i_weight
                hidden_x = input_x + hidden[max(j-1, 0), :][np.newaxis, :] @ h_weight + h_bias
                hidden_x = np.tanh(hidden_x)
                hidden[j, :] = hidden_x
                output_x = hidden_x @ o_weight + o_bias
                output[j, :] = output_x
            hiddens.append(hidden)
            outputs.append(output)
        return hiddens, outputs[-1]

    def backward(self, x, lr, grad, hiddens):
        for i in range(len(self.layers)):
            i_weight, h_weight, h_bias, o_weight, o_bias = self.layers[i].get_params()
            hidden = hiddens[i]
            next_h_grad = None
            i_weight_grad, h_weight_grad, h_bias_grad, o_weight_grad, o_bias_grad = [0] * 5

            for j in range(x.shape[0] - 1, -1, -1):
                out_grad = grad[j, :][np.newaxis, :]
                o_weight_grad += hidden[j, :][:, np.newaxis] @ out_grad
                o_bias_grad += out_grad
                h_grad = out_grad @ o_weight.T

                if j < x.shape[0] - 1:
                    hh_grad = next_h_grad @ h_weight.T
                    h_grad += hh_grad

                tanh_deriv = 1 - hidden[j][np.newaxis, :] ** 2
                h_grad = np.multiply(h_grad, tanh_deriv)
                next_h_grad = h_grad.copy()

                if j > 0:
                    h_weight_grad += hidden[j-1][:, np.newaxis] @ h_grad
                    h_bias_grad += h_grad

                i_weight_grad += x[j, :][:, np.newaxis] @ h_grad

            lr = lr / x.shape[0]
            i_weight -= i_weight_grad * lr
            h_weight -= h_weight_grad * lr
            h_bias -= h_bias_grad * lr
            o_weight -= o_weight_grad * lr
            o_bias -= o_bias_grad * lr
            self.layers[i].set_params([i_weight, h_weight, h_bias, o_weight, o_bias])
        return self.layers

    def train(self, train_set, valid_set):
        train_x, train_y = train_set
        valid_x, valid_y = valid_set

        loss_history = []
        for epoch in range(self.epochs):
            sequence_len = 7
            epoch_loss = 0
            for j in range(train_x.shape[0] - sequence_len):
                seq_x = train_x[j:(j+sequence_len), ]
                seq_y = train_y[j:(j+sequence_len), ]
                hiddens, outputs = self.forward(seq_x)
                grad = mse_grad(seq_y, outputs)
                self.layers = self.backward(seq_x, self.learning_rate, grad, hiddens)
                epoch_loss += mse(seq_y, outputs)
            loss_history.append(epoch_loss / len(train_x))

            if epoch % (self.epochs / 10) == 0:
                sequence_len = 7
                valid_loss = 0
                for j in range(valid_x.shape[0] - sequence_len):
                    seq_x = valid_x[j:(j+sequence_len), ]
                    seq_y = valid_y[j:(j+sequence_len), ]
                    _, outputs = self.forward(seq_x)
                    valid_loss += mse(seq_y, outputs)

                print(f"Epoch: {epoch} train loss {epoch_loss / len(train_x)} valid loss {valid_loss / len(valid_x)}")
        return loss_history
    
    def save_weights(self, path):
        weights_data = []
        for layer in self.layers:
            weights_data.append(layer.get_weights())

        file = open(path, "w")
        json.dump(weights_data, file)

    def load_weights(self, path):
        file = open(path, "r")
        weights_data = json.load(file)

        for i, layer_weights in enumerate(weights_data):
            self.layers[i].set_weights(layer_weights)