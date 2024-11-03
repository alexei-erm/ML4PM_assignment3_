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

FOLDER = 'models_baseline'    # CHANGE FOR EACH BRANCH ACCORDINGLY TO MODEL DURING TRAINING
LABELS = ['RUL']
XS_VAR = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
W_VAR = ['alt', 'Mach', 'TRA', 'T2']

folder = os.getcwd()
filename = f'{folder}/ncmapps_ds02.csv'

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        n_epochs=20,
        criterion=nn.MSELoss(),
        model_name='best_model_baseline',
        seed=42,
        device='cpu'
    ):
        self.seed = seed
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.bestmodel_cnt = 0
        self.losses4aggregation = {'train': [], 'eval': [], 'test': []}
        
        # adding time_stamp to model name to make sure the save models don't overwrite each other, 
        # you can customize your own model name with hyperparameters so that you can reload the model more easily
        time_stamp = time.strftime("%m%d%H%M%S")
        self.model_path = f'{FOLDER}/{model_name}_{time_stamp}.pt'
        # self.model_path = f'{folder}/models/{model_name}_{self.bestmodel_cnt}.pt'

        self.losses = {split: [] for split in ['train', 'eval', 'test']}
        
    def compute_loss(self, x, y, model=None):
        y = y.view(-1).to(self.device)
        x = x.to(self.device)  # Move x to device
        y_pred = self.model(x)  # Model is already on device
        y_pred = y_pred.view(-1)
        loss = self.criterion(y, y_pred)
        return loss, y_pred, y
    
    def train_epoch(self, loader):
        self.model.train()
        b_losses = []
        for x, y in loader:
            self.optimizer.zero_grad()
            # Remove these lines as data movement is handled in compute_loss
            # x.to(torch.device(self.device))
            # y.to(torch.device(self.device))
            
            loss, pred, target = self.compute_loss(x, y)
            loss.backward()
            self.optimizer.step()
            b_losses.append(loss.detach().cpu().numpy())  # Move to CPU for numpy
        
        agg_loss = np.sqrt((np.asarray(b_losses) ** 2).mean())
        self.losses['train'].append(agg_loss)
        return agg_loss

    # decorator, equivalent to with torch.no_grad():
    @torch.no_grad()
    def eval_epoch(self, loader, split='eval'):
        self.model.eval()
        b_losses = []
        for x, y in loader:
            loss, pred, target = self.compute_loss(x, y)
            b_losses.append(loss.detach().cpu().numpy())  # Add .cpu() here
        
        agg_loss = np.sqrt((np.asarray(b_losses) ** 2).mean())
        self.losses[split].append(agg_loss)
        return agg_loss
        
    def fit(self, loaders):
        print(f"Training model for {self.n_epochs} epochs...")
        train_loader, eval_loader, test_loader = loaders
        train_start = time.time()
        
        start_epoch = 0
        best_eval_loss = np.inf
            
        for epoch in range(start_epoch, self.n_epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            eval_loss = self.eval_epoch(eval_loader, split='eval')
            test_loss = self.eval_epoch(test_loader, split='test')

            # added :)
            self.losses4aggregation['train'].append(train_loss)
            self.losses4aggregation['eval'].append(eval_loss)
            self.losses4aggregation['test'].append(test_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.bestmodel_cnt += 1
                self.save(self.model, self.model_path)

            s = (
                f"[Epoch {epoch + 1}] "
                f"train_loss = {train_loss:.5f}, "
                f"eval_loss = {eval_loss:.5f}, "
                f"test_loss = {test_loss:.5f}"
            )

            epoch_time = time.time() - epoch_start
            s += f" [{epoch_time:.1f}s]"
            print(s)
    
        train_time = int(time.time() - train_start)
                
        print(f'Task done in {train_time}s')
    
    @ staticmethod
    def evaluate_model_performance(y_pred, y_true):
        """
        ## Task: Define Model Performance on RMSE and NASA score
        The performance of your implemented model should be evaluated using two common metrics applied in N-CMAPSS prognostics analysis:
        RMSE and NASA-score (in 1E5) as introduced in ["Fusing Physics-based and Deep Learning Models for Prognostics"](https://arxiv.org/abs/2003.00732)
        """
        # TODO: add implementation
        delta = y_true - y_pred
        score = np.sum(np.where(delta < 0, np.exp(-delta / 10.0), np.exp(delta / 13.0)))
        score = score / 1e5
        rmse = np.sqrt(np.mean(np.square(delta)))
        return score, rmse
    
    # decorator, equivalent to with torch.no_grad():
    @torch.no_grad()
    def eval_rul_prediction(self, test_loader):
        print(f"Evaluating test RUL...")
        
        ## MASK OUT EVAL and add explanation
        best_model = self.load(self.model) 
        best_model.to(self.device)
        best_model.eval()
        
        preds = []
        trues = []
        
        for x, y in tqdm(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            _, y_pred, y_target = self.compute_loss(x, y)
            preds.append(y_pred.detach().cpu().numpy())
            trues.append(y_target.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        df = pd.DataFrame({         
            'pred': preds,
            'true': trues,
            'err': np.sqrt((preds - trues)**2)
        })
        
        score, rmse = self.evaluate_model_performance(preds, trues)
        df_out = pd.DataFrame({
            'score': [score],
            'rmse': [rmse],
            'seed': [self.seed],
        })
        return df, df_out

    def save(self, model, model_path=None):
        os.makedirs(f'{folder}/models', exist_ok=True)
        if model_path is None:
            model_path = self.model_path 
        torch.save(model.state_dict(), model_path)
        
    def load(self, model, model_path=None):
        """
        loads the prediction model's parameters
        """
        if model_path is None:
            model_path = self.model_path
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model {model.__class__.__name__} saved in {model_path} loaded to {self.device}")
        return model

    def plot_losses(self):
        """
        :param losses: dict with losses
        """
        linestyles = {
            'train': 'solid', 
            'eval': 'dashed', 
            'test': 'dotted', 
        }
        for split, loss in self.losses.items():
            ls = linestyles[split]
            plt.plot(range(1, 1+len(loss)), loss, label=f'{split} loss', linestyle=ls)
            plt.yscale('log')
                
        plt.title("Training/Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
