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

visualizer = Visualizer(3, 2)

visualizer.create_chart_set(train_set, rnn, "Post-Training Test")
visualizer.create_chart_set(valid_set, rnn, "Validation Test")
visualizer.create_chart_set(test_set, rnn, "New Data Training Test")

visualizer.visualize()