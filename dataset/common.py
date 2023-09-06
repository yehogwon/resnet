from typing import Union, Callable, Optional

from torchvision.datasets import ImageNet, CIFAR10, CIFAR100

def create_dataset(name: str, root: str, train: bool, transform: Optional[Callable]) -> Union[ImageNet, CIFAR10, CIFAR100]: 
    if name not in ['imagenet', 'cifar10', 'cifar100']: 
        raise ValueError(f'Invalid dataset name: {name}')
    if name == 'imagenet': 
        return ImageNet(root=root, split='train' if train else 'val', transform=transform)
    elif name == 'cifar10':
        return CIFAR10(root=root, train=train, transform=transform)
    elif name == 'cifar100':
        return CIFAR100(root=root, train=train, transform=transform)
