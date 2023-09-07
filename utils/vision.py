from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

__all__ = [
    'imagenet_normalization_params',
    'cifar10_normalization_params',
    'cifar100_normalization_params'
]

def imagenet_normalization_params() -> Tuple[List[float], List[float]]: 
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def cifar10_normalization_params() -> Tuple[List[float], List[float]]:
    return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

def cifar100_normalization_params() -> Tuple[List[float], List[float]]:
    return [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

def calculate_normalization_params(train_loader: Optional[DataLoader], test_loader: Optional[DataLoader]) -> Tuple[List[float], List[float]]:
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
