import torch
import torch.nn as nn
import pandas as pd
from pre_proc import prepro
import numpy as np
from common.loader import CustomLoader as cl
from torch.utils.data import DataLoader
from common import nn_class

torch.manual_seed(1234)

def runPrepro():
    prepro.run()

def loadData(chemin, j): #Chemin = repertoire des datasets
    trainset = cl(chemin+'/train.csv', j)
    trainloader = DataLoader(dataset=trainset, batch_size=1, pin_memory=True,num_workers=4)
    validset = cl(chemin+'/vali.csv', j)
    validloader = DataLoader(dataset=validset, batch_size=1, pin_memory=True, num_workers=4)
    testset = cl(chemin+'/test.csv', j)
    testloader = DataLoader(dataset=testset, batch_size=1, pin_memory=True,num_workers=4)

    return trainloader, validloader, testloader

def train_CLASS():

    BATCH = 100
    EPOCHS = 100
    LR = 0.005

    trainloader, validloader, testloader = loadData('data/CLASS', 21)
    network = nn_class.Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    model = nn_class.train(network,optimizer, criterion, trainloader, validloader, testloader, EPOCHS) #implemnter le test

def train_NSP():
    pass

if __name__ == "__main__":
    # runPrepro()
    train_CLASS()