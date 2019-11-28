import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(21,5)
        self.fc2 = nn.Linear(5,10)
        # self.fc3 = nn.Linear(32,10)

        self.dropout = nn.Dropout(p=0.0)

    def forward(self, X):
        X = torch.relu(self.fc1(X))
        # X = F.tanh(self.fc2(X))
        X = F.softmax(self.fc2(X), dim = 1)

        return X


def train(network, optimizer, criterion, trainloader, validloader, testloader, EPOCHS):
    network.cuda()
    train_log=[]
    valid_log=[]
    for epoch in range(EPOCHS):
        training_loss = 0
        network.train() #Set the network to training mode
        for X, Y in trainloader:
            X=X.cuda()
            Y=Y.cuda()
            optimizer.zero_grad()
            out = network(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(trainloader) 
        validation_loss, accuracy = valid(network, criterion, validloader)

        log(EPOCHS, epoch, training_loss, validation_loss, accuracy)
        # last_loss = checkpoint(network, last_loss, validation_loss)

        train_log.append(training_loss)
        valid_log.append(validation_loss)

    show(train_log, valid_log)
    return network


def valid(network, criterion, validloader):
    validation_loss = 0
    accuracy = 0
    with torch.no_grad(): #Desactivate autograd engine (reduce memory usage and speed up computations)
        network.eval() #set the layers to evaluation mode(batchnorm and dropout)
        for X, Y in validloader:
            X=X.cuda()
            Y=Y.cuda()
            out = network(X)
            loss = criterion(out, Y)
            validation_loss += loss.item()


            predict = network(X)
            valuePred, indicePred = predict[0].max(0) #get the value and indice of the predicted output (Highest probability)
            _, indice = Y.max(1) #get the true indice so we could compare to the predicted one
            # _, predict_y = torch.max(predict, 1)

            # accuracy = accuracy + (torch.sum(Y==predict_y).float())
            # if indicePred==indice : accuracy += 1
        
        return validation_loss/len(validloader), 100*accuracy/(len(validloader)*len(Y))


def log(epochs, epoch, trainL, validL, accuracy):
    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
        "Training Loss: {:.3f}.. ".format(trainL),
        "Validation Loss: {:.3f}.. ".format(validL),
        "Validation Accuracy: {:.3f}".format(accuracy))

def show(trainL, validL):
    bestVali = min(validL)
    bestEpoch = validL.index(min(validL)) + 1
    plt.plot(trainL, label='Training loss')
    plt.plot(validL, label='Validation Loss')
    plt.title('Best validation perfomance is ' + str(bestVali) + ' at ' + str(bestEpoch))
    plt.legend(frameon=False)
    plt.show()