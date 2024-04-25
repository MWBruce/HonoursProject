import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

data = pd.read_csv('data.csv')

max_lags = 4
test_result = grangercausalitytests(data[['price', 'book value']], maxlags=max_lags, verbose=True)
