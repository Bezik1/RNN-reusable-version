import matplotlib.pyplot as plt
from helpers.functions import mse

class Visualizer:
    def __init__(self, size, columns) -> None:
        self.size = size
        self.columns = columns
        self.row = 1

    def draw(self, values_list, x_label, y_label, title):
        plt.subplot(self.size, self.columns, self.row)

        for values, label in values_list:
            plt.plot(values, label=label)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        self.row += 1
    
    def visualize(self):
        plt.tight_layout()
        plt.show()
    
    def create_chart_set(self, set, rnn, test_name):
        set_x, set_y = set
        train_hiddens, train_predictions = rnn.forward(set_x)
        train_history = [mse(y, pred) for y, pred in zip(set_x, train_predictions)]

        self.draw(
            [(train_history, "Cost Function")],
            "Sample",
            "Loss",
            "Cost Function"
        )

        self.draw(
            [(train_predictions, "Predictions"), (set_y, "Actual Value")],
            "Sample",
            "Price",
            test_name
        )