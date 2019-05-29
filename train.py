import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import json
import PIL
import argparse
import torch.utils.data 
import functioncaller

#Adding the argument parser for command line inputs
argument = argparse.ArgumentParser(description="train.py")

argument.add_argument('--arch', type=str, default='vgg16',)
argument.add_argument('--data_dir', type=str)
argument.add_argument('--learning_rate', type=float, default=0.001)
argument.add_argument('--gpu', action="store", default="gpu")
argument.add_argument('--hidden_units', type=int, default='120')
argument.add_argument('--save_dir', type=str, action="store")
argument.add_argument('--epochs', type=int, default=1)
argument.add_argument('--dropout', action="store", default=0.5)

#Parse Args
args = argument.parse_args()

datadir = args.data_dir
path = args.save_dir
lr = args.learning_rate
arch= args.arch
dropout = args.dropout
hidden_layer = args.hidden_units
epochs = args.epochs
power = args.gpu

train_loader, valid_loader, test_loader, train_data = functioncaller.load_transform_data(datadir)

model, optimizer, criterion = functioncaller.nsetup(arch,dropout,hidden_layer,lr, power)

functioncaller.train_network(model, criterion, optimizer, train_loader, valid_loader, epochs, 20, power, lr)

functioncaller.save_checkpoint(model, train_data, path,arch,hidden_layer,dropout,lr)