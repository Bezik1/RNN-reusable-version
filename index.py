from const.paths import HYPERPARAMETERS_PATH, TRAINED_NETWORK_PATH, STOCK_DATA_PATH
from helpers.Hyperparameters import Hyperparameters
from helpers.DataAnalyzer import DataAnalyzer
from helpers.functions import mse
from helpers.Visualizer import Visualizer
from structure.RNN import RNN

hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)

analyzer = DataAnalyzer()
train_set, valid_set, test_set = analyzer.analyze_stock(STOCK_DATA_PATH)

rnn = RNN(hyperparameters)
rnn.load_weights(TRAINED_NETWORK_PATH)

test_x, test_y = test_set
hiddens, predictions = rnn.forward(test_x)
loss_history = [mse(y, pred) for y, pred in zip(test_y, predictions)]

train_x, train_y = train_set
train_hiddens, train_predictions = rnn.forward(train_x)
train_history = [mse(y, pred) for y, pred in zip(train_x, train_predictions)]

valid_x, valid_y = valid_set
valid_hiddens, valid_predictions = rnn.forward(valid_x)
valid_history = [mse(y, pred) for y, pred in zip(valid_x, train_predictions)]

visualizer = Visualizer(2, 2)

visualizer.draw(
    [(loss_history, "Cost Function")],
    "Epoch",
    "Loss",
    "Cost Function"
)

visualizer.draw(
    [(train_predictions, "Predictions"), (train_y, "Actual Value")],
    "X",
    "Y",
    "Post-Training"
)

visualizer.draw(
    [(predictions, "Predictions"), (test_y, "Actual Value")],
    "X",
    "Y",
    "Test"
)

visualizer.draw(
    [(valid_predictions, "Predictions"), (valid_y, "Actual Value")],
    "X",
    "Y",
    "Valid Test"
)

visualizer.visualize()