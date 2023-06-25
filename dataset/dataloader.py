import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def make_dataloader(path: str, mode: str, batch_size: int): 
    if mode == 'train': 
        dataset = CIFAR10(root=os.path.join(path, 'cifar10/train'), train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        ]))
    elif mode == 'test': 
        dataset = CIFAR10(root=os.path.join(path, 'cifar10/test'), train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
        ]))
    else: 
        raise ValueError('mode should be either train or test')
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader
