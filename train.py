import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset.dataloader import make_dataloader
import model.resnet as resnet
from model.parallel import ParallelWrapper

import wandb
from tqdm import tqdm
import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_data, test_data, optimizer, loss, epoch, device, ckpt_path):
    save_term = epoch // 100
    model_name = str(model)
    prefix_ = f'{model_name}-{timestamp}'
    for epoch_idx in tqdm(range(epoch)): 
        model.train()
        total_loss = 0
        total_acc = 0
        for batch_idx, (x, y) in enumerate(train_data): 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss_ = loss(y_hat, y)
            loss_.backward()
            optimizer.step()
            total_loss += loss_.item()
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            wandb.log({'epoch': epoch_idx})
        wandb.log({'train/loss': total_loss / len(train_data), 'train/acc': total_acc / len(train_data)})
        test_loss, test_acc = evaluate(model, test_data, loss, device)
        wandb.log({'test/loss': test_loss, 'test/acc': test_acc})
        if (epoch_idx + 1) % save_term == 0: 
            torch.save(model.state_dict(), os.path.join(ckpt_path, f'{prefix_}-{epoch_idx + 1}.pth'))
    
def evaluate(model, test_data, loss, device): 
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad(): 
        for batch_idx, (x, y) in enumerate(test_data): 
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            total_loss += loss(y_hat, y).item()
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
    return total_loss / len(test_data), total_acc / len(test_data)

def main(args): 
    train_data = make_dataloader(path=args.dataset, mode='train', batch_size=args.batch_size)
    test_data = make_dataloader(path=args.dataset, mode='test', batch_size=args.batch_size)
    
    model_name = args.model.lower()
    assert model_name in ['resnet18', 'resnet34'], 'model should be either resnet18 or resnet34'
    if model_name == 'resnet18':
        model_ = resnet.resnet18()
    elif model_name == 'resnet34':
        model_ = resnet.resnet34()
    if torch.cuda.device_count() > 1:
        model = ParallelWrapper(model_).to(device)
    else: 
        model = model_.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss()

    train_info = {
        'name': model_name, 
        'epoch': args.epoch,
        'batch_size': args.batch_size,
        'learning rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay, 
        'device': device
    }
    
    wandb.init(project='resnet-demo', config=train_info)
    print('\n'.join([f' - {k}: {v}' for k, v in train_info.items()]))
    
    train(model, train_data, test_data, optimizer, loss, args.epoch, device, args.ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='resnet18 or resnet34 is supported at the moment.')
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--epoch', type=int, default=600000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()
    main(args)
    