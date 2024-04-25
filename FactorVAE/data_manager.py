import pandas as pd

data_path = 'data.csv'
data_df = pd.read_csv(data_path)

save_dir = './data'
data_df.to_pickle(f"{save_dir}/train_csi300.pkl")
data_df.to_pickle(f"{save_dir}/valid_csi300.pkl")
data_df.to_pickle(f"{save_dir}/test_csi300.pkl")