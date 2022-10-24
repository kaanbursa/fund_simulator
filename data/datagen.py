import os,sys,inspect
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from datetime import datetime
import pandas as pd
from utils.indicators import indicator_list

import numpy as np
from data.preprocessing import data_split, get_price, DataProcessor

snpnas = pd.read_csv("../datasets/snpnasstocks.csv")


sn_tickers = snpnas.Symbol.unique()


if __name__ == "__main__":
    stock_loader = DataProcessor('stocks', sn_tickers, '2000-01-01', '2022-07-01')
    stock_loader.preprocess_for_train('snpnasdaqprices', False, [], '2000-01-01')
