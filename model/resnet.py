from typing import Optional, Type, Union, List

import torch
import torch.nn as nn

def conv3x3(in_channels: int, out_channels: int, stride: int=1, padding: int=1) -> nn.Conv2d: 
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride, 
        padding=padding, 
        bias=False
    )

def conv1x1(in_channels: int, out_channels: int, stride: int=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module): 
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            stride: int=1, 
            downsample: Optional[nn.Module]=None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)

        z = self.conv2(z)
        z = self.bn2(z)

        if self.downsample is not None: 
            x = self.downsample(x)

        z += x
        z = self.relu(z)

        return z

# TODO: bottleneck block support
class Bottleneck(nn.Module): 
    pass
