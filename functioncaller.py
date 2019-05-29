import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import json
import PIL
from PIL import Image

import argparse

arch = {"vgg16" : 25088,
        "alexnet" : 9216,
        "densenet121" : 1024}

def load_transform_data(datadir = "./ImageClassifier/flowers"):
    data_dir = datadir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
                  transforms.RandomRotation(20),
                  transforms.RandomResizedCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Load the dataset with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    #Define the dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    return train_loader, valid_loader, test_loader, train_data

# Define the architecture and return criterion, optimizer & model
def nsetup (model_name = 'vgg16', dropout = 0.5, hidden_layer = 120, lr = 0.001, power='gpu'):

    if model_name == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained = True)
    else:
        print("Oh no you have chosen {} which is not a valid model".format(model_name))
   
    for param in model.parameters():
        param.requires_grad = False
    
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(arch[model_name], hidden_layer)),
            ('relu1', nn.ReLU()),
            ('hidden_layer', nn.Linear(hidden_layer, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))

        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        if torch.cuda.is_available() and power == 'gpu':
            model.cuda()

        return model, criterion, optimizer
    
def train_network(model, criterion, optimizer, train_loader, valid_loader, epochs=1, print_every=1, power='gpu', lr=0.001):
    steps = 0
    running_loss = 0

    # change to cuda
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda')
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )

    for e in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            steps += 1
            optimizer.zero_grad()
            if torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
    
            # Forward and backward passes
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
            
                for i, (inputs2,labels2) in enumerate(valid_loader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(valid_loader)
                accuracy = accuracy /len(valid_loader)
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Lost {:.4f}".format(vlost),
                   "Accuracy: {:.4f}".format(accuracy))
            
                running_loss = 0

def save_checkpoint(model, train_data, path='./ImageClassifier/checkpoint.pth',arch ='vgg16', hidden_layer=120,dropout=0.5, lr=0.001, epochs=12):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
                  'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'hidden_layer': hidden_layer,
                  'dropout': dropout,
                  'lr': lr,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'hidden_units': 4096,
                  'path' : path
                  }
    
    torch.save(checkpoint, path)
    
def load_checkpoint(path='./ImageClassifier/checkpoint.pth'):
    
    checkpoint = torch.load(path)
    arch = checkpoint['arch']
    hidden_layer = checkpoint['hidden_layer']
    dropout = checkpoint['dropout']
    lr = checkpoint['lr']
    model,_,_ = nsetup(arch, dropout, hidden_layer, lr)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    
def process_image(image_path):
    print(str(image_path))
    im = Image.open(image_path)
    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image_new = process(im)
    return image_new

def predict(image_path, model, topk=5, power='gpu'):
  
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')
        
    image_trained = Image.open(str(image_path))
    image_trained = process_image(image_trained)
    image_trained = image_trained.unsqueeze_(0)
    image_trained = image_trained.float()
       
    with torch.no_grad():
        output = model.forward(image_trained.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)   