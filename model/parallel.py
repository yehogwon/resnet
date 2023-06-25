from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.module import Module

class ParallelWrapper(nn.Module): 
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = nn.DataParallel(model)
    
    def forward(self, *args, **kwargs): 
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name: str): 
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
