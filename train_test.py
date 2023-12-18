from const.paths import HYPERPARAMETERS_PATH, TRAINED_NETWORK_PATH, STOCK_DATA_PATH
from const.visualizer import VISUALIZER_COLUMNS, VISUALIZER_SIZE
from helpers.Hyperparameters import Hyperparameters
from helpers.DataAnalyzer import DataAnalyzer
from helpers.Visualizer import Visualizer
from helpers.functions import mse
from structure.RNN import RNN

hyperparameters = Hyperparameters(HYPERPARAMETERS_PATH)

analyzer = DataAnalyzer()
#analyzer.transform_intraday(STOCK_DATA_PATH)
train_set, valid_set, test_set = analyzer.analyze_stock(STOCK_DATA_PATH)

rnn = RNN(hyperparameters)
loss_history = rnn.train(train_set, valid_set)
rnn.save_weights(TRAINED_NETWORK_PATH)

visualizer = Visualizer(3, 2)

visualizer.create_chart_set(train_set, rnn, "Post-Training Test")
visualizer.create_chart_set(valid_set, rnn, "Validation Test")
visualizer.create_chart_set(test_set, rnn, "New Data Training Test")

visualizer.visualize()