from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

def calculate_normalisation_params(train_loader: Optional[DataLoader], test_loader: Optional[DataLoader]) -> Tuple[List[float], List[float]]:
    """
    Calculate the mean and standard deviation of each channel
    for all observations in training and test datasets. The
    results can then be used for normalisation
    """ 
    chan0 = np.array([])
    chan1 = np.array([])
    chan2 = np.array([])
    
    if train_loader is not None: 
        for images, _ in train_loader:
            chan0 = np.concatenate((chan0, images[:, 0, :, :].cpu().flatten()))
            chan1 = np.concatenate((chan0, images[:, 1, :, :].cpu().flatten()))
            chan2 = np.concatenate((chan0, images[:, 2, :, :].cpu().flatten()))
    
    if test_loader is not None:
        for images, _ in test_loader:
            chan0 = np.concatenate((chan0, images[:, 0, :, :].cpu().flatten()))
            chan1 = np.concatenate((chan0, images[:, 1, :, :].cpu().flatten()))
            chan2 = np.concatenate((chan0, images[:, 2, :, :].cpu().flatten()))
        
    means = [np.mean(chan0), np.mean(chan1), np.mean(chan2)]
    stds  = [np.std(chan0), np.std(chan1), np.std(chan2)]
    
    return means, stds
