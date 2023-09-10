from typing import Optional, Type, Union, List

import torch
import torch.nn as nn

__all__ = [
    'resnet18', 
    'resnet20',
    'resnet32',
    'resnet34',
    'resnet44',
    'resnet50',
    'resnet56',
    'resnet101',
    'resnet110',
    'resnet152'
    'resnet1202',
]

basic_block_config = {
    'resnet18': [2, 2, 2, 2],
    'resnet20': [3, 3, 3, 3],
    'resnet32': [5, 5, 5, 5],
    'resnet44': [7, 7, 7, 7],
    'resnet56': [9, 9, 9, 9],
    'resnet110': [18, 18, 18, 18],
    'resnet1202': [200, 200, 200, 200]
}

''' The implementation below follows a GitHub repository: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py 
    For more details, refer to the original paper: https://arxiv.org/abs/1512.03385 '''

def _short_cut(in_channels: int, out_channels: int, stride: int, expansion: int) -> Optional[nn.Module]:
    if stride != 1 or in_channels != expansion * out_channels: 
        return nn.Sequential(
            nn.Conv2d(in_channels, expansion * out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(expansion * out_channels)
        )
    else: 
        return None

class BasicBlock(nn.Module): 
    expansion = 1 # do nothing, but for compatibility with Bottleneck

    def __init__(self, in_channels: int, out_channels: int, stride=1) -> None: 
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = _short_cut(in_channels, out_channels, stride, self.expansion)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        z += self.shortcut(x) if self.shortcut is not None else x
        return self.relu(z)
    
class Bottleneck(nn.Module): 
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride=1) -> None: 
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = _short_cut(in_channels, out_channels, stride, self.expansion)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.relu(self.bn2(self.conv2(z)))
        z = self.bn3(self.conv3(z))
        z += self.shortcut(x) if self.shortcut is not None else x
        return self.relu(z)

BlockType = Union[BasicBlock, Bottleneck]
RESNET_IN_CHANNELS = 64
RESNET_KERNEL_SIZE = 7

class ResNet(nn.Module): 
    def __init__(self, block: BlockType, n_blocks: List[int], n_classes: int) -> None: 
        super().__init__()

        self.in_channels = RESNET_IN_CHANNELS

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=RESNET_KERNEL_SIZE, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.layers1 = self._make_layer(block, 64, n_blocks[0], stride=1)
        self.layers2 = self._make_layer(block, 128, n_blocks[1], stride=2)
        self.layers3 = self._make_layer(block, 256, n_blocks[2], stride=2)
        self.layers4 = self._make_layer(block, 512, n_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        self.relu = nn.ReLU()
    
    def _make_layer(self, block: BlockType, channels: int, n_blocks: int, stride: int) -> nn.Sequential: 
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides: 
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.layers1(z)
        z = self.layers2(z)
        z = self.layers3(z)
        z = self.layers4(z)
        z = self.gap(z)
        z = z.view(z.shape[0], -1)
        return self.fc(z)

''' BasicBlock '''
def resnet18(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes)

def resnet20(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [3, 3, 3, 3], n_classes)

def resnet32(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [5, 5, 5, 5], n_classes)

def resnet34(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], n_classes)

def resnet44(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [7, 7, 7, 7], n_classes)

def resnet56(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [9, 9, 9, 9], n_classes)

def resnet110(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [18, 18, 18, 18], n_classes)

def resnet1202(n_classes: int) -> ResNet:
    return ResNet(BasicBlock, [200, 200, 200, 200], n_classes)

''' Bottleneck '''
def resnet50(n_classes: int) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], n_classes)

def resnet101(n_classes: int) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], n_classes)

def resnet152(n_classes: int) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], n_classes)
