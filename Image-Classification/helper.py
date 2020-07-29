#Importing the libraries
import time
import torch
import json
import numpy as np
import argparse, sys
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models

data_dir = 'flowers'
#save the paths of all the three directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

data_transforms = {'train_transform':train_transform,'valid_transform':valid_transform,'test_transform':test_transform}

#Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform = train_transform)
test_data = datasets.ImageFolder(test_dir,transform = test_transform)
valid_data = datasets.ImageFolder(valid_dir,transform = valid_transform)
image_datasets = {'train_data':train_data,'valid_data':valid_data,'test_data':test_data}

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data,batch_size = 64,shuffle = True )
testloader = torch.utils.data.DataLoader(test_data,batch_size = 64,shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data,batch_size = 64,shuffle = True)
dataloaders = {'trainloader':trainloader,'validloader':validloader,'testloader':testloader}
