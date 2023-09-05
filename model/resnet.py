import torch
import torch.nn as nn

# TODO: bottleneck support (three convs in a row)
# TODO: check if bias is needed in convs
class ResidualBlock(nn.Module): 
    def __init__(self, n_filters: int, subsample: bool=False) -> None:
        '''
        :param n_filters: 
        :param subsample: True if the first convolution should downsample the input ()
        '''
        super().__init__()

        s = 0.5 if subsample else 1.0

        self.conv1 = nn.Conv2d(int(n_filters * s), n_filters, kernel_size=3, stride=int(1 / s), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv2d(int(n_filters * s), n_filters, kernel_size=1, stride=int(1 / s), bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)   
    
    def shortcut(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor: 
        if x.shape != z.shape: # projection only if needed
            d = self.downsample(x)
            p = torch.zeros_like(z)
            return z + torch.cat((d, p), dim=1)
        else: 
            return z + x
    
    def forward(self, x: torch.Tensor, shortcuts: bool=False) -> torch.Tensor: 
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)

        z = self.conv2(z)
        z = self.bn2(z)

        if shortcuts:
            z = self.shortcut(z, x)
        z = self.relu2(z)

        return z

class ResNet(nn.Module): 
    def __init__(self, n: 
