import os
import time 
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler

LABELS = ['RUL']
XS_VAR = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
W_VAR = ['alt', 'Mach', 'TRA', 'T2']

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SlidingWindowDataset(Dataset):
    def __init__(self, dataframe, window=50, stride=1, horizon=1, device='cpu'):
        """Sliding window dataset with RUL label

        Args:
            dataframe (pd.DataFrame): dataframe containing scenario descriptors and sensor reading
            window (int, optional): sequence window length. Defaults to 50.
            stride (int, optional): data stride length. Defaults to 1.
            horizon (int, optional): prediction forcasting length. Defaults to 1.
        """
        self.window = window
        self.stride = stride
        self.horizon = horizon
        self.device = device 
        
        self.X = np.array(dataframe[XS_VAR+W_VAR].values).astype(np.float32)
        self.y = np.array(dataframe['RUL'].values).astype(np.float32)
        if 'ds' in dataframe.columns:
            unqiue_cycles = dataframe[['ds', 'unit', 'cycle']].value_counts(sort=False)
        else:
            unqiue_cycles = dataframe[['unit', 'cycle']].value_counts(sort=False)
        self.indices = torch.from_numpy(self._get_indices(unqiue_cycles)).to(device)

    # TODO add comment
    def _get_indices(self, unqiue_cycles):
        cycles = unqiue_cycles.to_numpy()
        idx_list = []
        for i, c_count in enumerate(cycles):
            c_start = sum(cycles[:i])
            c_end = c_start + (c_count - self.window - self.horizon)
            if c_end + self.horizon < len(self.X): # handling y not in the last seq case
                idx_list += [_ for _ in np.arange(c_start, c_end + 1, self.stride)]
        return np.asarray([(idx, idx+self.window) for idx in idx_list])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        i_start, i_stop = self.indices[i]
        x = self.X[i_start:i_stop, :]
        y = self.y[i_start]
        x = x.permute(1, 0)
        return x, y
    
def create_datasets(df, window_size, train_units, test_units, device='cpu'):
    df_train = df[df['unit'].isin(train_units)]
    train_dataset = SlidingWindowDataset(df_train, window=window_size)

    df_test = df[df['unit'].isin(test_units)]    
    test_dataset = SlidingWindowDataset(df_test, window=window_size)

    # normalizing features
    scaler = MinMaxScaler()
    train_dataset.X = scaler.fit_transform(train_dataset.X)
    test_dataset.X = scaler.transform(test_dataset.X)

    # convert numpy array to tensors
    datasets = [train_dataset, test_dataset]
    for d in datasets:
        d.X = torch.from_numpy(d.X).to(device)
        d.y = torch.from_numpy(d.y).to(device)
    
    return datasets

def create_data_loaders(datasets, batch_size=256, val_split=0.2):
    # fixed seed for data splits for reproducibility
    random.seed(0)
    np.random.seed(0)
    
    d_train, d_test = datasets
    dataset_size = len(d_train)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(d_train, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(d_train, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)      

    d_info = f"train_size: {len(train_indices)}\t"
    d_info += f"validation_size: {len(val_indices)}\t"
    d_info += f"test_size: {len(d_test)}"
    print(d_info)
    return train_loader, val_loader, test_loader


def plot_aggregated_losses(train_losses, eval_losses, test_losses):
    epochs = range(len(train_losses[0]))
    
    def plot_with_variance(data, title, ylabel):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean, label=title)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        plt.title(f'{title} with Variance')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_with_variance(train_losses, 'Train Loss', 'Loss')
    plot_with_variance(eval_losses, 'Eval Loss', 'Loss')
    plot_with_variance(test_losses, 'Test Loss', 'Loss')
