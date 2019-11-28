import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from math import sin
from loader import CustomLoader as cl
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,16)
        self.fc3 = nn.Linear(16,32)
        self.fc4 = nn.Linear(32,16)
        self.fc5 = nn.Linear(16,8)
        self.fc6 = nn.Linear(8,1)

    def forward(self, X):
        X = (self.fc1(X))
        X = (self.fc2(X))
        X = (self.fc3(X))
        X = (self.fc4(X))
        X = (self.fc5(X))

        X = self.fc6(X)

        return X



def log(epochs, epoch, trainL, testL):#Add the accuracy
    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
        "Training Loss: {:.3f}.. ".format(trainL),
        "Testing Loss: {:.3f}.. ".format(testL))

def show(net):
    x = np.arange(0, 14, 0.2)
    y = [sin(a) for a in x]

    pred=[]
    test = cl(x, y)
    test = DataLoader(dataset=test, batch_size=1)
    for X,Y in test:
        pred.append(net(X))
        print('X :',X, 'Y :',Y, 'PRED :',net.forward(X))

    # pred = [net(torch.Tensor(a)) for a in x]
    # pred = net(torch.Tensor(list(x)))

    pred = [float(p) for p in pred]
    print(pred)
    plt.scatter(x,pred, marker="o", color="red")
    plt.plot(x,y, color="blue")
    # plt.show()