import pandas as pd

data = pd.read_csv("data.csv")

size = len(data)

train_size = int(size*.6)
val_size = int(size *.2)
test_size = size - train_size - val_size

train_data = data.loc[:train_size]
val_data = data.loc[train_size:train_size+val_size]
test_data = data.loc[train_size+val_size:]

train_data.to_pickle("train_df.pkl")
test_data.to_pickle("test_df.pkl")
val_data.to_pickle("val_df.pkl")