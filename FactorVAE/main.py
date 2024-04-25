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
from tqdm.auto import tqdm
import argparse
from module import FactorVAE, FeatureExtractor, FactorDecoder, FactorEncoder, FactorPredictor, AlphaLayer, BetaLayer
from dataset import StockDataset
from train_model import train, validate, test
from utils import set_seed, DataArgument
import wandb

parser = argparse.ArgumentParser(description='Train a FactorVAE model on stock data')

parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--num_latent', type=int, default=17, help='number of variables')
parser.add_argument('--num_portfolio', type=int, default=17, help='number of stocks')
parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
parser.add_argument('--num_factor', type=int, default=48, help='number of factors')
parser.add_argument('--hidden_size', type=int, default=20, help='hidden size')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--run_name', type=str, help='name of the run')
parser.add_argument('--save_dir', type=str, default='./best_models', help='directory to save model')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
parser.add_argument('--normalize', action='store_true', help='whether to normalize the data')
args = parser.parse_args()

data_args = DataArgument(use_qlib=False, normalize=True, select_feature=False)

assert args.seq_len == data_args.seq_len, "seq_len in args and data_args must be the same"
# assert args.normalize == data_args.normalize, "normalize in args and data_args must be the same"
        
train_df = pd.read_pickle('./data/train_df.pkl')
valid_df = pd.read_pickle('./data/val_df.pkl')
test_df = pd.read_pickle('./data/test_df.pkl')
    
    # wandb.log({"train_df": train_df, "valid_df": valid_df, "test_df": test_df})


def main(args, data_args):
    
    set_seed(args.seed)
    # make directory to save model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # create model
    feature_extractor = FeatureExtractor(num_latent=args.num_latent, hidden_size=args.hidden_size)
    factor_encoder = FactorEncoder(num_factors=args.num_factor, num_portfolio=args.num_portfolio, hidden_size=args.hidden_size)
    alpha_layer = AlphaLayer(args.hidden_size)
    beta_layer = BetaLayer(args.hidden_size, args.num_factor)
    factor_decoder = FactorDecoder(alpha_layer, beta_layer)
    factor_predictor = FactorPredictor(args.batch_size, args.hidden_size, args.num_factor)
    factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
    
    # create dataloaders
    train_ds = StockDataset(train_df, args.batch_size, args.seq_len)
    valid_ds = StockDataset(valid_df, args.batch_size, args.seq_len)
    
    train_dataloader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=300, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # def save_data(dataloader, file_path):
    #     with open(file_path, 'w') as f:
    #         # first_batch = next(iter(dataloader))
    #         # features, labels = first_batch
    #         for batch_index, (features, labels) in enumerate(dataloader):
    #             for i in range(features.size(0)):

    #                 feature_tensor = features[i]
    #                 label_tensor = labels[i]


    #                 # f.write(f'Feature Dimensions: {feature_tensor.size()}\n')
    #                 for row in feature_tensor:
    #                     row_str = ', '.join(f'{x:.4f}' for x in row.tolist())
    #                     f.write(f'{row_str}\n')


    #                 # label_str = ', '.join(str(x) for x in label_tensor.tolist())
    #                 # f.write(f'Label: [{label_str}]\n')

    # save_data(train_dataloader, 'sampletest.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*************** Using {device} ***************")
    args.device = device
        
    factorVAE.to(device)
    best_val_loss = 1000000.0
    optimizer = torch.optim.Adam(factorVAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs)
    
    if args.wandb:
        wandb.init(project="FactorVAE", config=args, name=f"{args.run_name}")
        wandb.config.update(args)

    # Start Trainig
    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(factorVAE, train_dataloader, optimizer, args)
        val_loss = validate(factorVAE, valid_dataloader, args)
        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #? save model in save_dir
            
            #? torch.save
            save_root = os.path.join(args.save_dir, f'saved_model.pt')
            torch.save(factorVAE.state_dict(), save_root)
            
        if args.wandb:
            wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss}) 
    
    if args.wandb:
        wandb.log({"Best Validation Loss": best_val_loss})
        wandb.finish()
    
if __name__ == '__main__':
    main(args, data_args)
