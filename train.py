import argparse

import torch

class Trainer: 
    pass

def main(args: argparse.Namespace) -> None: 
    pass

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train a ResNet model on a variety of datasets.')
    parser.add_argument('--name', type=str, help='Name of the experiment.')

    parser.add_argument('--dataset', type=str, choices=['imagenet', 'cifar10', 'cifar100'], help='Dataset to train on.')
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Model to train, one of the following: resnet18, resnet34, resnet50, resnet101, resnet152.')
    
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for training.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum to use for training.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay to use for training.')
    
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')

    args = parser.parse_args()
    main(args)
