import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module): 
    def __init__(self, dim: int, sub: bool) -> None: # dim_change is True when this block is the first block of a stage
        super().__init__()

        scale = 2 if sub else 1
        self.conv1 = nn.Conv2d(in_channels=int(dim / scale), out_channels=dim, kernel_size=3, stride=scale, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        
        self.relu = nn.ReLU()
        self._dim_change = nn.Conv2d(in_channels=int(dim / scale), out_channels=dim, kernel_size=1, stride=2, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
    
    def shortcut(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.shape != z.shape:
            x = self._dim_change(x)
        return x + z
    
    def forward(self, x: torch.Tensor, shortcut: bool=True) -> torch.Tensor: 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if shortcut: 
            out = self.shortcut(x, out)
        return self.relu(out)

# FIXME: Bottleneck has tons of bugs, so not being used at the moment
class Bottleneck(nn.Module): 
    def __init__(self, dim: int, first: bool=False, second: bool=False) -> None: 
        super().__init__()

        # FIRST SECOND    SCALE
        #   0     0   -> scale = 1
        #   0     1   -> scale = 4
        #   1     0   -> scale = 0.5
        #   1     1   -> FORBIDDEN

        both = first and second
        assert not both, 'one of first and second should be True, not both'

        if first: 
            scale = 0.5
        elif second: 
            scale = 4
        else:
            scale = 1

        self.conv1 = nn.Conv2d(in_channels=int(dim * scale), out_channels=dim, kernel_size=1, stride=scale, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(num_features=dim * 4)
        
        self.relu = nn.ReLU()
        self._dim_change = nn.Conv2d(in_channels=int(dim * scale), out_channels=dim * 4, kernel_size=1, stride=scale, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('!!! forward !!!') 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        print('bef x:', x.shape, ', out:', out.shape)
        x = self._dim_change(x)
        print('aft x:', x.shape, ', out:', out.shape)
        
        return self.relu(x + out)

class ResNet(nn.Module): 
    def __init__(self, sizes: list[int], bottleneck: bool=True) -> None:
        super().__init__()
        
        assert type(sizes) == list and len(sizes) == 4, 'sizes should be a list of 4 integers'
        assert not bottleneck, 'bottleneck is not supported yet'

        self.sizes = sizes
        self.bottleneck = bottleneck

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()

        # BlockClass = Bottleneck if bottleneck else Block
        BlockClass = Block

        self.module_stack = [
            nn.Sequential(*[BlockClass(dim=64, sub=False) for i in range(sizes[0])]), 
            nn.Sequential(*[BlockClass(dim=128, sub=i==0) for i in range(sizes[1])]), 
            nn.Sequential(*[BlockClass(dim=256, sub=i==0) for i in range(sizes[2])]),
            nn.Sequential(*[BlockClass(dim=512, sub=i==0) for i in range(sizes[3])])
        ]

        # self.module_stack = [
        #     nn.Sequential(*[BlockClass(dim=2 ** (i + 6), dim_change=i==0 and not (d == 0)) for i in range(sizes[d])])
        # for d in range(4)]

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=1000, bias=True)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.relu(self.bn(self.pool(self.conv(x))))
        out = self.module_stack[0](out)
        out = self.module_stack[1](out)
        out = self.module_stack[2](out)
        out = self.module_stack[3](out)
        out = self.gap(out)
        out = self.fc(out.view(out.size(0), -1))
        return out
    
    def __repr__(self):
        return f'ResNet-{sum(self.sizes) * (3 if self.bottleneck else 2) + 2}'

def resnet18() -> ResNet:
    return ResNet([2, 2, 2, 2])

def resnet34() -> ResNet:
    return ResNet([3, 4, 6, 3])
