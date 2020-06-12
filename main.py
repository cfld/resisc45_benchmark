#!/usr/bin/env python

"""
    main.py
"""

import sys
import json
import argparse
import numpy as np
from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True

from torchvision import transforms, datasets

def set_seeds(seed):
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed + 111)
    _ = np.random.seed(seed + 222)

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--init-lr',    type=float, default=0.1)
    parser.add_argument('--seed',       type=int,   default=789)
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

# --
# Dataset statistics

stats = {
    "mean" : (0.3680, 0.3810, 0.3436),
    "std"  : (0.2035, 0.1854, 0.1849),
}

# --
# IO

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.75, 1), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    
    transforms.ToTensor(),
    transforms.Normalize(stats['mean'], stats['std']),
])

transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(stats['mean'], stats['std']),
])

dataset_train = datasets.ImageFolder(root='./data/NWPU-RESISC45_split/train', transform=transform_train)
dataset_valid = datasets.ImageFolder(root='./data/NWPU-RESISC45_split/valid', transform=transform_test)
assert dataset_train.classes == dataset_valid.classes

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

dataloader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=2 * args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

# --
# Model definition
# Derived from models in `https://github.com/kuangliu/pytorch-cifar`

class Model(nn.Module):
    def train_epoch(self, loader, sched=None):
        _ = model.train()
        
        n_correct, n_total = 0, 0
        
        gen = tqdm(loader, total=len(loader))
        for x, y in gen:
            x, y = x.cuda(), y.cuda()
            
            out  = self.forward(x)
            loss = F.cross_entropy(out, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            pred = out.argmax(axis=-1)
            n_correct += int((y == pred).sum())
            n_total   += int(pred.shape[0])
            
            gen.set_postfix(**{
                "loss" : float(loss), 
                "acc"  : float(n_correct / n_total)
            })
            
            if sched is not None:
                sched.step()
        
        return {
            "acc" : float(n_correct / n_total)
        }
    
    def eval_epoch(self, loader, n_batches=np.inf):
        _   = model.eval()
        
        n_correct, n_total = 0, 0
        
        gen = tqdm(loader, total=min(len(loader), n_batches))
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(gen):
                if batch_idx == n_batches: break
                
                out  = self.forward(x.cuda())
                
                pred = out.argmax(axis=-1).cpu()
                n_correct += int((y == pred).sum())
                n_total   += int(pred.shape[0])
                
                gen.set_postfix(**{
                    "acc" : float(n_correct / n_total)
                })
        
        return {
            "acc" : float(n_correct / n_total)
        }


class PreActBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
            
    def forward(self, x):
        out      = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out      = self.conv1(out)
        out      = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet18(Model):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__()
        
        self.in_channels = 64
        
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layers = nn.Sequential(
            self._make_layer(64,   64, num_blocks[0], stride=1),
            self._make_layer(64,  128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 512, num_blocks[3], stride=2),
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.prep(x)
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    


# --
# Define model

num_classes = len(dataset_train.classes)

model = ResNet18(num_classes=num_classes).cuda()
opt   = torch.optim.SGD(model.parameters())

sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, 
    max_lr=args.init_lr, 
    epochs=args.epochs, 
    steps_per_epoch=len(dataloader_train)
)

t = time()
for epoch in range(args.epochs):
    train = model.train_epoch(dataloader_train, sched)
    valid = model.eval_epoch(dataloader_valid, n_batches=100)
    print(json.dumps({
        "epoch"     : int(epoch),
        "train_acc" : float(train['acc']),
        "valid_acc" : float(valid['acc']),
        "lr"        : sched.get_lr()[0],
        "time"      : time() - t,
    }))
    sys.stdout.flush()
    
valid = model.eval_epoch(dataloader_valid)