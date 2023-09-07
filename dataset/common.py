from typing import Union, Callable, Optional

from torchvision.datasets import ImageNet, CIFAR10, CIFAR100

def create_dataset(name: str, root: str, split: str, transform: Optional[Callable]=None) -> Union[ImageNet, CIFAR10, CIFAR100]: 
    if name not in ['imagenet', 'cifar10', 'cifar100']: 
        raise ValueError(f'Invalid dataset name: {name}')
    if split not in ['train', 'val', 'trainval']:
        raise ValueError(f'Invalid split name: {split}')
    
    if split == 'trainval': 
        return create_dataset(name, root, 'train', transform=transform) + create_dataset(name, root, 'val', transform=transform)
    
    if name == 'imagenet': 
        return ImageNet(root=root, split=split, transform=transform)
    elif name == 'cifar10':
        return CIFAR10(root=root, train=split == 'train', transform=transform)
    elif name == 'cifar100':
        return CIFAR100(root=root, train=split == 'train', transform=transform)

def get_class_counts(dataset: Union[ImageNet, CIFAR10, CIFAR100]) -> int: 
    if isinstance(dataset, ImageNet): 
        return 1000
    elif isinstance(dataset, CIFAR10): 
        return 10
    elif isinstance(dataset, CIFAR100): 
        return 100
    else: 
        raise ValueError(f'Invalid dataset type: {type(dataset)}')
