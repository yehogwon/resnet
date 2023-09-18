import os
import argparse
from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

import wandb
from tqdm import tqdm

from dataset.common import create_dataset, get_class_counts
from utils.vision import *
from resnet import ResNet, BasicBlock, Bottleneck

class Trainer: 
    def __init__(
            self, 
            project_name: str,
            exp_name: str, 
            dataset: str, 
            transform: Optional[Callable], 
            model: nn.Module, 
            ckpt_path: str, 
            ckpt_interval: int, 
            device: str='cpu'
    ) -> None:
        self.project_name = project_name
        self.exp_name = exp_name
        self.dataset_name = dataset
        self.transform = transform
        self.train_dataset = create_dataset(dataset, os.path.join('dataset', self.dataset_name), split='train', transform=transform)
        self.val_dataset = create_dataset(dataset, os.path.join('dataset', self.dataset_name), split='val', transform=transform)

        self.model = model
        self.ckpt_path = ckpt_path
        self.ckpt_interval = ckpt_interval
        self.device = device
    
    # TODO: learning rate scheduler
    def train(
            self, 
            batch_size: int, 
            n_epoch: int, 
            start_epoch: int, 
            lr: float, 
            weight_decay: float, 
            betas: Tuple[float, float], 
            scheduler: Optional[str]=None,
            scheduler_args: Optional[dict]=None,
            wandb_log: bool=True
    ) -> None: 
        if wandb_log: 
            wandb.init(
                project=self.project_name,
                name=f'{self.model}-{self.dataset_name}-{self.exp_name}', 
                config={
                    'dataset': self.dataset_name,
                    'device': self.device,
                    'batch_size': batch_size,
                    'n_epoch': n_epoch,
                    'lr': lr,
                    'weight_decay': weight_decay, 
                    'betas': betas, 
                    'lr_scheduler': scheduler,
                    'lr_scheduler_args': scheduler_args
                }
            )
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        if scheduler == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_args['cosine']['T_max'], eta_min=scheduler_args['cosine']['eta_min'], last_epoch=start_epoch)
        elif scheduler == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_args['step']['step_size'], gamma=scheduler_args['step']['gamma'], last_epoch=start_epoch)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.model = self.model.to(self.device)

        for epoch in range(start_epoch, n_epoch): 
            train_acc, train_loss = self._train_iteration(train_loader, loss_fn, optimizer, desc=f'Epoch {epoch}/{n_epoch}', wandb_log=wandb_log)

            log_info = {
                'epoch': epoch, 
                'learning_rate': lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else lr,
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
            
            if lr_scheduler is not None:
                lr_scheduler.step()
    
    def _train_iteration(
            self, 
            dataloader: DataLoader, 
            loss_fn: nn.Module, 
            optimizer: optim.Optimizer, 
            desc: str='Training', 
            wandb_log: bool=True
    ) -> Tuple[float, float]:
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
        
        return n_correct / len(dataloader.dataset), sum(losses) / len(losses) # acc, loss_avg

    def validate(
            self, 
            batch_size: int
    ) -> Tuple[float, float]: 
        test_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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

        acc = n_correct / len(self.val_dataset)
        loss_avg = sum(losses) / len(losses)
        return acc, loss_avg

    def _save_model(self, name: str) -> str:
        ckpt_saved = os.path.join(self.ckpt_path, name)
        torch.save(self.model.state_dict(), ckpt_saved)
        return ckpt_saved # return the path to saved checkpoint

def main(args: argparse.Namespace) -> None: 
    n_classes = get_class_counts(args.dataset)
    if args.model == 'resnet18': 
        model = ResNet(BasicBlock, [2, 2, 2, 2], n_classes=n_classes)
    elif args.model == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], n_classes=n_classes)
    elif args.model == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], n_classes=n_classes)
    elif args.model == 'resnet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3], n_classes=n_classes)
    elif args.model == 'resnet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3], n_classes=n_classes)

    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model))
    
    # get dataset mean and std
    if args.dataset == 'imagenet':
        mean, std = imagenet_normalization_params()
    elif args.dataset == 'cifar10':
        mean, std = cifar10_normalization_params()
    elif args.dataset == 'cifar100':
        mean, std = cifar100_normalization_params()

    if args.dataset == 'imagenet': 
        transform = transforms.Compose([
            transforms.Resize(256), #  In the original paper, the images are resized with their shortest side randomly sampled in [256, 480] for scale augmentation.
            transforms.RandomCrop(244), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.dataset in ['cifar10', 'cifar100']: 
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    trainer = Trainer(
        project_name=args.project_name,
        exp_name=args.exp_name,
        dataset=args.dataset,
        transform=transform,
        model=model, 
        ckpt_path=args.ckpt_path,
        ckpt_interval=args.ckpt_interval,
        device=args.device
    )

    trainer.train(
        batch_size=args.batch_size,
        n_epoch=args.epochs,
        start_epoch=args.start_epoch, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        scheduler=args.lr_scheduler,
        scheduler_args={
            'cosine': {
                'T_max': args.cosine_t_max,
                'eta_min': args.cosine_eta_min
            },
            'step': {
                'step_size': args.step_step_size,
                'gamma': args.step_gamma
            }
        },
        wandb_log=args.wandb
    )

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train a ResNet model on a variety of datasets.')
    parser.add_argument('--project_name', type=str, default='ResNet', help='Name of the project.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment.')

    parser.add_argument('--dataset', type=str, required=True, choices=['imagenet', 'cifar10', 'cifar100'], help='Dataset to train on.')
    parser.add_argument('--model', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Model to train, one of the following: resnet18, resnet34, resnet50, resnet101, resnet152.')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to save checkpoints.')
    parser.add_argument('--ckpt-interval', type=int, default=10, help='Interval to save checkpoints at.')

    parser.add_argument('--pretrained-model', type=str, help='path to pretrained model')
    parser.add_argument('--start-epoch', type=int, default=1, help='start epoch for training')
    
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for training.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay to use for training.')

    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer (momentum).')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer.')

    parser.add_argument('--lr-scheduler', type=str, required=False, choices=['cosine', 'step'], help='Learning rate scheduler to use for training.')

    parser.add_argument('--cosine-t-max', type=int, default=100, help='Period for cosine learning rate scheduler.')
    parser.add_argument('--cosine-eta-min', type=float, default=0.0, help='Minimum learning rate for cosine learning rate scheduler.')

    parser.add_argument('--step-step-size', type=int, default=30, help='Period for step learning rate scheduler to decay learning rate.')
    parser.add_argument('--step-gamma', type=float, default=0.1, help='Multiplicative factor for step learning rate scheduler.')
    
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')

    args = parser.parse_args()
    main(args)
