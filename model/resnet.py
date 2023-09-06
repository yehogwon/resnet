from typing import Optional, Type, Union, List

import torch
import torch.nn as nn

class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self.prj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.bn(self.prj(x))

class BasicBlock(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels: 
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else: 
            self.shortcut = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        return self.relu(z + self.shortcut(x))
