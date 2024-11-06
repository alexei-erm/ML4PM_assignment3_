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


def plot_aggregated_losses(train_losses, eval_losses, test_losses, model_name='baseline', save=False, show=True):
    epochs = range(len(train_losses[0]))
    
    def plot_with_variance(data, title, ylabel, filename):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean, label=title)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        if save:
            os.makedirs('plots_training', exist_ok=True)
            plt.savefig(f'plots_training/{filename}')
        if show:
            plt.show()
        else:
            plt.close()
    
    plot_with_variance(train_losses, 'Train Loss', 'RMSE Loss', f'{model_name}_trainloss.png')
    plot_with_variance(eval_losses, 'Eval Loss', 'RMSE Loss', f'{model_name}_evalloss.png')
    plot_with_variance(test_losses, 'Test Loss', 'RMSE Loss', f'{model_name}_testloss.png')


def plot_test_rul_predictions(df_list, model_name='baseline', save=False, show=True):
    for i, df_test_out in enumerate(df_list):
        plt.figure(figsize=(10, 6))
        df_test_out.plot(y=['true', 'pred'], ax=plt.gca())
        plt.title(f'Test RUL Predictions - Run {i}')
        plt.xlabel('Index')
        plt.ylabel('RUL')
        plt.legend(['True RUL', 'Predicted RUL'])
        plt.grid(True)
        if save:
            os.makedirs('plots_RUL', exist_ok=True)
            plt.savefig(f'plots_RUL/{model_name}_testRUL_{i}.png')
        if show:
            plt.show()
        else:
            plt.close()

