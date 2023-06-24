import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module): 
    def __init__(self, dim: int, dim_change: bool=False) -> None: # dim_change is True when this block is the first block of a stage
        super().__init__()

        scale = 2 if dim_change else 1
        self.conv1 = nn.Conv2d(in_channels=int(dim / scale), out_channels=dim, kernel_size=3, stride=scale, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        
        self.relu = nn.ReLU()
        self._dim_change = nn.Conv2d(in_channels=int(dim / scale), out_channels=dim, kernel_size=1, stride=2, bias=False) if dim_change else nn.Identity()

        # TODO: Initialize weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        x = self._dim_change(x)
        
        return self.relu(x + out)

class Bottleneck(nn.Module): 
    def __init__(self, dim: int, dim_change: bool=False) -> None: # dim_change is True when this block is the first block of a stage
        super().__init__()

        scale = 2 if dim_change else 1
        self.conv1 = nn.Conv2d(in_channels=int(dim / scale), out_channels=dim, kernel_size=1, stride=scale, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=dim)
        
        self.relu = nn.ReLU()
        self._dim_change = nn.Conv2d(in_channels=int(dim / scale), out_channels=dim, kernel_size=1, stride=2, bias=False) if dim_change else nn.Identity()

        # TODO: Initialize weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        x = self._dim_change(x)
        
        return self.relu(x + out)

# TODO: Implement ResNet module (this is not perfect yet)
class ResNet(nn.Module): 
    def __init__(self, sizes: list[int], bottleneck: bool=True) -> None:
        super().__init__()
        
        assert type(sizes) == list and len(sizes) == 4, 'sizes should be a list of 4 integers'

        self.sizes = sizes
        self.bottleneck = bottleneck

        # TODO: the value of paddings is not specified in the paper

        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()

        BlockClass = Bottleneck if bottleneck else Block

        self.module_stack = [
            nn.ModuleList([Block(dim=16, dim_change=False) for i in range(n)]), 
            nn.ModuleList([Block(dim=32, dim_change=i==0) for i in range(n)]), 
            nn.ModuleList([Block(dim=64, dim_change=i==0) for i in range(n)])
        ]

        self.gap = nn.AdaptiveAvgPool2d(output_size=1000)
        self.fc = nn.Linear(in_features=1000, out_features=10, bias=True)

        # TODO: Initialize weights

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.relu(self.bn1(self.conv1(x)))
        for ml in self.module_stack: 
            for m in ml: 
                out = m(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def __repr__(self):
        return f'ResNet-{sum(self.sizes) * (3 if self.bottleneck else 2) + 2}'
