from typing import Type, Union, List

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

class Bottleneck(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels: 
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else: 
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.relu(self.bn2(self.conv2(z)))
        z = self.bn3(self.conv3(z))
        return self.relu(z + self.shortcut(x))

class ResNet(nn.Module): 
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], n_blocks: List[int], n_classes: int) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        n_channels = [64, 128, 256, 512]
        
        self.layer1 = [block(n_channels[0], n_channels[0], stride=1)]
        self.layer1 += [block(n_channels[0], n_channels[0], stride=1)] * (n_blocks[0] - 1)
        
        self.layer2 = [block(n_channels[0], n_channels[1], stride=2)]
        self.layer2 += [block(n_channels[1], n_channels[1], stride=1)] * (n_blocks[1] - 1)

        self.layer3 = [block(n_channels[1], n_channels[2], stride=2)]
        self.layer3 += [block(n_channels[2], n_channels[2], stride=1)] * (n_blocks[2] - 1)

        self.layer4 = [block(n_channels[2], n_channels[3], stride=2)]
        self.layer4 += [block(n_channels[3], n_channels[3], stride=1)] * (n_blocks[3] - 1)

        self.layers = nn.ModuleList(self.layer1 + self.layer2 + self.layer3 + self.layer4)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.maxpool(self.bn(self.conv(x)))

        for layer in self.layers:
            z = layer(z)
            print(z.shape)

        z = self.gap(z)
        z = z.view(z.size(0), -1)
        return self.fc(z)
