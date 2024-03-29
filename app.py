import torch
import torch.nn as nn
import pandas as pd
from pre_proc import prepro
import numpy as np
from common.loader import CustomLoader as cl
from torch.utils.data import DataLoader
from common import nn_class, nn_nsp

torch.manual_seed(1234)


def runPrepro():
    prepro.run()

def loadData(chemin, j): #Chemin = repertoire des datasets
    trainset = cl(chemin+'/train.csv', j)
    trainloader = DataLoader(dataset=trainset, batch_size=16)
    validset = cl(chemin+'/vali.csv', j)
    validloader = DataLoader(dataset=validset, batch_size=16)
    testset = cl(chemin+'/test.csv', j)
    testloader = DataLoader(dataset=testset, batch_size=1)

    return trainloader, validloader, testloader

def train_CLASS():
    EPOCHS = 601
    LR = 0.00003

    trainloader, validloader, testloader = loadData('data/CLASS', 20)
    network = nn_class.Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    model = nn_class.train(network,optimizer, criterion, trainloader, validloader, testloader, EPOCHS) 

def train_NSP():
    EPOCHS = 3000
    LR = 0.00003 #retry with 0.0001 and 0.0005

    trainloader, validloader, testloader = loadData('data/NSP', 20)
    network = nn_nsp.Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    model = nn_nsp.train(network,optimizer, criterion, trainloader, validloader, testloader, EPOCHS) 

if __name__ == "__main__":
    # runPrepro()
    # train_CLASS()
    train_NSP()
