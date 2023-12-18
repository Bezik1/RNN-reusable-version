from const.analyze import PREDICTORS, TARGET, STOCK_PREDICTORS, STOCK_TAREGT
import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self) -> None:
        pass

    def analyze(self, path):
        data = pd.read_csv(path, index_col=0)
        data = data.ffill()

        data["tmax"].head(100)

        mean_values = np.mean(data[PREDICTORS], axis=0)
        std_dev_values = np.std(data[PREDICTORS], axis=0)

        data[PREDICTORS] = (data[PREDICTORS] - mean_values) / std_dev_values

        split_data = np.split(data, [int(.7*len(data)), int(.85*len(data))])
        train_set, valid_set, test_set = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data]
        
        return train_set, valid_set, test_set
    
    def analyze_stock(self, path):
        data = pd.read_csv(path, index_col=0)
        data = data.ffill()

        data["open"].head(3500)

        mean_values = np.mean(data[STOCK_PREDICTORS], axis=0)
        std_dev_values = np.std(data[STOCK_PREDICTORS], axis=0)

        data[STOCK_PREDICTORS] = (data[STOCK_PREDICTORS] - mean_values) / std_dev_values

        split_data = np.split(data, [int(.7*len(data)), int(.85*len(data))])
        train_set, valid_set, test_set = [[d[STOCK_PREDICTORS].to_numpy(), d[[STOCK_TAREGT]].to_numpy()] for d in split_data]
        
        return train_set, valid_set, test_set

    def transform_intraday(self, path):
        data = pd.read_csv(path)
        data = data.ffill()

        data['next_open'] = data['open'].shift(1)
        data.to_csv(path, index=False)