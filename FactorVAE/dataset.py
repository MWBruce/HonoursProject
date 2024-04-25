import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import datetime
import random
from tqdm.auto import tqdm

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, df, num_stock, sequence_length):
        self.df = df
        self.num_stock = num_stock
        self.sequence_length = sequence_length
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='s')
        
        self.df = self.df.dropna(subset=['price'])
        
        self.df = pd.get_dummies(self.df, columns=['sector'], prefix='sector')
        self.expected_sectors = [f'sector_{i}' for i in range(1, 12)]
        for column in self.expected_sectors:
            if column not in self.df.columns:
                self.df[column] = 0

        self.instrument_groups = self.df.groupby('ticker')
        self.group_indices = []
        for name, group in self.instrument_groups:
            indices = [(name, i) for i in range(0,len(group) - self.sequence_length + 1,sequence_length)]
            self.group_indices.extend(indices)

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx):
        ticker, group_start_idx = self.group_indices[idx]
        group = self.instrument_groups.get_group(ticker)

        group = group.fillna(method='ffill').fillna(0)
        
        data = group.iloc[group_start_idx:group_start_idx + self.sequence_length]
        
        input_data = data.drop(['price', 'ticker', 'timestamp'] + self.expected_sectors, axis=1).values
        label = data['price'].values
        sector_data = data[self.expected_sectors].values

        input_data = np.hstack([input_data, sector_data])

        return input_data, label