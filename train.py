import os
import argparse
from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

from dataset.common import create_dataset, get_class_counts

class Trainer: 
    def __init__(self, exp_name: str, dataset: str, transform: Optional[Callable], model: nn.Module, ckpt_path: str, ckpt_interval: int, device: str='cpu') -> None:
        self.exp_name = exp_name
        self.dataset_name = dataset
        self.transform = transform
        self.train_dataset = create_dataset(dataset, train=True, transform=transform)
        self.test_dataset = create_dataset(dataset, train=False, transform=transform)

        self.model = model
        self.ckpt_path = ckpt_path
        self.ckpt_interval = ckpt_interval
        self.device = device
    
    def train(self, batch_size: int, n_epoch: int, lr: float, weight_decay: float, wandb_log: bool=True) -> None: 
        if wandb_log: 
            wandb.init(
                project=f'{self.model}-{self.dataset_name}-{self.exp_name}', 
                config={
                    'dataset': self.dataset_name,
                    'device': self.device,
                    'batch_size': batch_size,
                    'n_epoch': n_epoch,
                    'lr': lr,
                    'weight_decay': weight_decay
                }
            )
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.model = self.model.to(self.device)

        for epoch in range(1, n_epoch): 
            train_loss, train_acc = self._train_iteration(train_loader, loss_fn, optimizer, desc=f'Epoch {epoch}/{n_epoch}', wandb_log=wandb_log)

            log_info = {
                'epoch': epoch, 
                'train_loss': train_loss, 
                'train_acc': train_acc
            }

            val_acc, val_loss = self.validate(batch_size) # run validation only on CIFAR-10/100
            log_info.update({
                'val_loss': val_loss, 
                'val_acc': val_acc
            })

            print(' | '.join([f'{k}: {v}' for k, v in log_info.items()]))
            if wandb_log: 
                wandb.log(log_info)

            if epoch % self.ckpt_interval == 0:
                ckpt_path = self._save_model(f'{self.exp_name}_{epoch}.pth')
                print(f'Checkpoint saved: {ckpt_path}')
    
    def _train_iteration(self, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, desc: str='Training', wandb_log: bool=True) -> Tuple[float, float]:
        losses = []
        n_correct = 0
        
        self.model.train()
        for x, y in tqdm(dataloader, desc=desc):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            y_pred = self.model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            n_correct += (F.softmax(y_pred, dim=1).argmax(dim=1) == y).sum().item()
            losses.append(loss.item())

            if wandb_log: 
                wandb.log({'loss': loss.item()})
            
            del x, y, y_pred, loss
        
        return n_correct / len(self.train_dataset), sum(losses) / len(losses) # acc, loss_avg

    def validate(self, batch_size: int) -> Tuple[float, float]: 
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        loss_fn = nn.CrossEntropyLoss()
        n_correct = 0
        losses = []
        
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f'Validation'):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                n_correct += (F.softmax(y_pred, dim=1).argmax(dim=1) == y).sum().item()

                loss = loss_fn(y_pred, y)
                losses.append(loss.item())

                del x, y, y_pred, loss

        acc = n_correct / len(self.test_dataset)
        loss_avg = sum(losses) / len(losses)
        return acc, loss_avg

    def _save_model(self, name: str) -> str:
        ckpt_saved = os.path.join(self.ckpt_path, name)
        torch.save(self.model.state_dict(), ckpt_saved)
        return ckpt_saved # return the path to saved checkpoint

def main(args: argparse.Namespace) -> None: 
    pass

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train a ResNet model on a variety of datasets.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment.')

    parser.add_argument('--dataset', type=str, required=True, choices=['imagenet', 'cifar10', 'cifar100'], help='Dataset to train on.')
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Model to train, one of the following: resnet18, resnet34, resnet50, resnet101, resnet152.')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to save checkpoints.')
    parser.add_argument('--ckpt-interval', type=int, default=10, help='Interval to save checkpoints at.')
    
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for training.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum to use for training.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay to use for training.')
    
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')

    args = parser.parse_args()
    main(args)