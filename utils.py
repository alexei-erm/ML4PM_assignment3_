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

from model import CNN


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

def plot_test_rul_predictions(df_list, df_all, model_name, save=False, show=True):
    """
    Plot test RUL predictions for the best performing run based on RMSE.
    
    Args:
        df_list (list): List of DataFrames containing RUL predictions
        df_all (pd.DataFrame): DataFrame containing performance metrics for each run
        model_name (str): Name of the model for saving plots
        save (bool): Whether to save the plots to disk
        show (bool): Whether to display the plots
    """
    # Find the best performing run based on RMSE
    best_idx = df_all['rmse'].idxmin()
    best_rmse = df_all.loc[best_idx, 'rmse']
    best_score = df_all.loc[best_idx, 'score']
    
    # Get the corresponding predictions
    df_test_out = df_list[best_idx]
    
    plt.figure(figsize=(12, 6))
    df_test_out.plot(y=['true', 'pred'], ax=plt.gca())
    plt.title(f'RUL predictions over test units\nBest Run (RMSE: {best_rmse:.2f}, Score: {best_score:.2f})')
    plt.xlabel('Index')
    plt.ylabel('RUL')
    plt.legend(['True RUL', 'Predicted RUL'])
    plt.grid(True)
    
    if save:
        os.makedirs('plots_RUL', exist_ok=True)
        plt.savefig(f'plots_RUL/{model_name}_best_testRUL.png')
    if show:
        plt.show()
    else:
        plt.close()


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

def evaluate_per_unit_stored(df_all, all_df_test, test_units=[11, 14, 15]):
    """
    Evaluate per-unit performance for test units using stored predictions from the best model run.
    Handles variable lengths for different units.
    """
    # Find best model based on RMSE
    best_idx = df_all['rmse'].idxmin()
    df_pred = all_df_test[best_idx]
    
    # Find transitions between units by looking for jumps in RUL values
    # RUL typically resets to a higher value when switching to a new unit
    rul_diff = df_pred['true'].diff()
    unit_change_indices = [0] + list(np.where(rul_diff > 10)[0]) + [len(df_pred)]
    
    # Calculate metrics for each test unit
    results = []
    
    for i, unit in enumerate(test_units):
        start_idx = unit_change_indices[i]
        end_idx = unit_change_indices[i + 1]
        
        unit_df = df_pred.iloc[start_idx:end_idx]
        delta = unit_df['true'] - unit_df['pred']
        score = np.sum(np.where(delta < 0, 
                              np.exp(-delta / 10.0), 
                              np.exp(delta / 13.0))) / 1e5
        rmse = np.sqrt(np.mean(np.square(delta)))
        
        results.append({
            'unit': str(unit),
            'type': 'Test',
            'rmse': rmse,
            'nasa_score': score,
            'samples': len(unit_df)
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate test averages
    test_metrics = results_df.agg({
        'rmse': 'mean',
        'nasa_score': 'sum',
        'samples': 'sum'
    })
    
    # Add average row
    avg_row = pd.DataFrame([{
        'unit': 'Avg',
        'type': 'Summary',
        'rmse': test_metrics['rmse'],
        'nasa_score': test_metrics['nasa_score'],
        'samples': test_metrics['samples']
    }])
    
    results_df = pd.concat([results_df, avg_row], ignore_index=True)
    
    # Print table
    print("\nPer-unit performance metrics:")
    print("=" * 80)
    print(f"{'Unit':>6} {'Type':>8} {'RMSE':>12} {'NASA Score':>12} {'Samples':>12}")
    print("-" * 80)
    
    # Print test units and average
    for _, row in results_df.iterrows():
        print(f"{row['unit']:>6} {row['type']:>8} {row['rmse']:12.3f} {row['nasa_score']:12.3f} {row['samples']:>12.0f}")
    
    print("=" * 80)
    
    return results_df, best_idx