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

def init_weights(m):
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()

class CNN(nn.Module):
    '''
    A 1D-CNN model that is based on the paper "Fusing physics-based and deep learning models for prognostics"
    from Manuel Arias Chao et al. (with batchnorm layers)
    '''
    
    def __init__(self, 
                 in_channels=18, 
                 out_channels=1,
                 window=50, 
                 n_ch=10, 
                 n_k=10, 
                 n_hidden=50, 
                 n_layers=3,
                 dropout=0.,
                 padding='same',
                 use_batchnorm = False):
        """
        Args:
            n_features (int, optional): number of input features. Defaults to 18.
            window (int, optional): sequence length. Defaults to 50.
            n_ch (int, optional): number of channels (filter size). Defaults to 10.
            n_k (int, optional): kernel size. Defaults to 10.
            n_hidden (int, optional): number of hidden neurons for regressor. Defaults to 50.
            n_layers (int, optional): number of convolution layers. Defaults to 5.
            use_batchnorm (bool): whether to use batch normalization. Defaults to False.
        """
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        # List to store convolutional layers
        layers = []
        
        # First Conv Layer
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=n_ch, kernel_size=n_k, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(n_ch))
        layers.append(nn.ReLU())
        
        # Additional Conv Layers
        for _ in range(1, n_layers):
            layers.append(nn.Conv1d(in_channels=n_ch, out_channels=n_ch, kernel_size=n_k, padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(n_ch))
            layers.append(nn.ReLU())
        
        # Combine the convolution layers into a sequential block
        self.conv_layers = nn.Sequential(*layers)
        
        # Flatten layer before passing to fully connected layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers (regressor)
        self.fc1 = nn.Linear(n_ch * window, n_hidden)
        self.dropout = nn.Dropout(dropout)
        if use_batchnorm:
            self.bn_fc = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, out_channels)
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, x):
        # TODO: implement forward pass
        # Apply convolution layers
        x = self.conv_layers(x)
        
        # Flatten the output from convolution layers
        x = self.flatten(x)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
