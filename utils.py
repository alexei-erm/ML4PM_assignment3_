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


# def evaluate_best_model_per_unit(df_all, test_loader, model_name='baseline', device='cpu', test_units=[11, 14, 15]):
#     """
#     Find the best model based on df_all metrics, load it, and evaluate performance on each test unit.
    
#     Args:
#         df_all (pd.DataFrame): DataFrame containing performance metrics for each model run
#         test_loader (DataLoader): DataLoader containing test data
#         model_name (str): Base name of the model (e.g., 'baseline')
#         device (str): Device to run evaluation on
#         test_units (list): List of test unit numbers to evaluate separately
        
#     Returns:
#         pd.DataFrame: DataFrame containing RMSE and NASA score for each unit
#         int: Index of the best model
#     """
#     # Find best model based on RMSE
#     best_idx = df_all['rmse'].idxmin()
#     best_rmse = df_all.loc[best_idx, 'rmse']
#     best_score = df_all.loc[best_idx, 'score']
#     print(f"Loading best model (idx={best_idx}) with RMSE={best_rmse:.3f}, Score={best_score:.3f}")
    
#     # Get all model files in the directory
#     model_dir = f'models_{model_name}'
#     model_files = sorted([f for f in os.listdir(model_dir) if f.startswith(model_name)])
    
#     if not model_files:
#         raise ValueError(f"No model files found in {model_dir} with prefix {model_name}")
    
#     # Load the best model
#     model_path = os.path.join(model_dir, model_files[best_idx])
#     print(f"Loading model from: {model_path}")
    
#     model = CNN().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     # Get predictions in batches
#     all_preds = []
#     all_trues = []
    
#     with torch.no_grad():
#         for x, y in test_loader:
#             x = x.to(device)
#             y = y.to(device)
#             pred = model(x).cpu().numpy().flatten()
#             true = y.cpu().numpy().flatten()
#             all_preds.extend(pred)
#             all_trues.extend(true)
    
#     # Convert to numpy arrays
#     all_preds = np.array(all_preds)
#     all_trues = np.array(all_trues)
    
#     # Split predictions by unit
#     samples_per_unit = len(all_preds) // len(test_units)
    
#     # Calculate metrics for each unit
#     results = []
#     for i, unit in enumerate(test_units):
#         start_idx = i * samples_per_unit
#         end_idx = (i + 1) * samples_per_unit if i < len(test_units) - 1 else len(all_preds)
        
#         pred = all_preds[start_idx:end_idx]
#         true = all_trues[start_idx:end_idx]
        
#         # Calculate metrics
#         delta = true - pred
#         score = np.sum(np.where(delta < 0, 
#                               np.exp(-delta / 10.0), 
#                               np.exp(delta / 13.0))) / 1e5
#         rmse = np.sqrt(np.mean(np.square(delta)))
        
#         results.append({
#             'unit': unit,
#             'rmse': rmse,
#             'nasa_score': score,
#             'samples': len(pred)
#         })
    
#     results_df = pd.DataFrame(results)
    
#     # Add average row
#     avg_row = pd.DataFrame([{
#         'unit': 'Average',
#         'rmse': results_df['rmse'].mean(),
#         'nasa_score': results_df['nasa_score'].mean(),
#         'samples': results_df['samples'].sum()
#     }])
#     results_df = pd.concat([results_df, avg_row], ignore_index=True)
    
#     return results_df, best_idx

# def evaluate_best_model_per_unit(df_all, test_loader, model_name='baseline', device='cpu', test_units=[11, 14, 15]):
#     """
#     Evaluate the best model using the same evaluation method as during training.
    
#     Args:
#         df_all (pd.DataFrame): DataFrame containing performance metrics for each model run
#         test_loader (DataLoader): DataLoader containing test data
#         model_name (str): Base name of the model
#         device (str): Device to run evaluation on
#         test_units (list): List of test unit numbers to evaluate separately
#     """
#     # Find best model based on RMSE
#     best_idx = df_all['rmse'].idxmin()
#     best_rmse = df_all.loc[best_idx, 'rmse']
#     best_score = df_all.loc[best_idx, 'score']
#     print(f"Loading best model (idx={best_idx}) with RMSE={best_rmse:.3f}, Score={best_score:.3f}")
    
#     # Get model path
#     model_dir = f'models_{model_name}'
#     model_files = sorted([f for f in os.listdir(model_dir) if f.startswith(model_name)])
#     model_path = os.path.join(model_dir, model_files[best_idx])
#     print(f"Loading model from: {model_path}")
    
#     # Load model
#     model = CNN().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     # Get predictions using same method as trainer
#     criterion = nn.MSELoss()
#     all_preds = []
#     all_trues = []
    
#     with torch.no_grad():
#         for x, y in test_loader:
#             x = x.to(device)
#             y = y.to(device)
#             y_pred = model(x)
#             y_pred = y_pred.view(-1)
#             loss = criterion(y, y_pred)
#             all_preds.append(y_pred.detach().cpu().numpy())
#             all_trues.append(y.detach().cpu().numpy())
    
#     # Concatenate all predictions and true values
#     all_preds = np.concatenate(all_preds)
#     all_trues = np.concatenate(all_trues)
    
#     # Calculate metrics for each unit
#     results = []
#     samples_per_unit = len(all_preds) // len(test_units)
    
#     for i, unit in enumerate(test_units):
#         start_idx = i * samples_per_unit
#         end_idx = (i + 1) * samples_per_unit if i < len(test_units) - 1 else len(all_preds)
        
#         unit_preds = all_preds[start_idx:end_idx]
#         unit_trues = all_trues[start_idx:end_idx]
        
#         # Calculate metrics using same method as trainer
#         delta = unit_trues - unit_preds
#         score = np.sum(np.where(delta < 0, 
#                               np.exp(-delta / 10.0), 
#                               np.exp(delta / 13.0))) / 1e5
#         rmse = np.sqrt(np.mean(np.square(delta)))
        
#         results.append({
#             'unit': unit,
#             'rmse': rmse,
#             'nasa_score': score,
#             'samples': len(unit_preds)
#         })
    
#     results_df = pd.DataFrame(results)
    
#     # Calculate overall metrics
#     delta_all = all_trues - all_preds
#     overall_score = np.sum(np.where(delta_all < 0, 
#                                   np.exp(-delta_all / 10.0), 
#                                   np.exp(delta_all / 13.0))) / 1e5
#     overall_rmse = np.sqrt(np.mean(np.square(delta_all)))
    
#     # Add average row
#     avg_row = pd.DataFrame([{
#         'unit': 'Average',
#         'rmse': overall_rmse,  # Using overall metrics instead of mean of unit metrics
#         'nasa_score': overall_score,
#         'samples': results_df['samples'].sum()
#     }])
#     results_df = pd.concat([results_df, avg_row], ignore_index=True)
    
#     # Verify against df_all
#     print("\nVerification:")
#     print(f"Original best model metrics - RMSE: {best_rmse:.3f}, Score: {best_score:.3f}")
#     print(f"Re-evaluated metrics     - RMSE: {overall_rmse:.3f}, Score: {overall_score:.3f}")
    
#     return results_df, best_idx

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