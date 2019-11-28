import pandas as pd
from pre_proc import prepro
import numpy as np
from common.loader import CustomLoader as cl
from torch.utils.data import DataLoader

def runPrepro():
    prepro.run()

def loadData(chemin, j): #Chemin = repertoire des datasets
    trainset = cl(chemin+'/train.csv', j)
    trainloader = DataLoader(dataset=trainset, batch_size=1)
    validset = cl(chemin+'/vali.csv', j)
    validloader = DataLoader(dataset=validset, batch_size=1)
    testset = cl(chemin+'/test.csv', j)
    testloader = DataLoader(dataset=testset, batch_size=1)

    return trainloader, validloader, testloader

def train_CLASS():
    pass

def train_NSP():
    pass

if __name__ == "__main__":
    runPrepro()