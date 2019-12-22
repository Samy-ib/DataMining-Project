import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_on_gpu = torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20,320)
        self.fc2 = nn.Linear(320,640)
        self.fc3 = nn.Linear(640,800)
        self.fc4 = nn.Linear(800,640)
        self.fc5 = nn.Linear(640,320)
        self.fc6 = nn.Linear(320,120)
        self.fc7 = nn.Linear(120,3)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, X):
        X = self.dropout(F.gelu(self.fc1(X)))
        X = self.dropout(F.gelu(self.fc2(X)))
        X = self.dropout(F.gelu(self.fc3(X)))
        X = self.dropout(F.gelu(self.fc4(X)))
        X = self.dropout(F.gelu(self.fc5(X)))
        X = self.dropout(F.gelu(self.fc6(X)))


        X = F.softmax(self.fc7(X), dim = 1)

        return X


def train(network, optimizer, criterion, trainloader, validloader, testloader, EPOCHS):
    if train_on_gpu :
        network.cuda()
    train_log=[]
    valid_log=[]
    best_loss = 0.04

    for epoch in range(EPOCHS):
        training_loss = 0
        network.train() #Set the network to training mode
        for X, Y in trainloader:
            if train_on_gpu:
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

        if validation_loss<best_loss:
            torch.save(network.state_dict(), 'models/nsp_model.pt')
            best_loss=validation_loss

        log(EPOCHS, epoch, training_loss, validation_loss, accuracy)
        # last_loss = checkpoint(network, last_loss, validation_loss)

        train_log.append(training_loss)
        valid_log.append(validation_loss)
    test(network, testloader)
    show(train_log, valid_log)
    return network


def valid(network, criterion, validloader):
    validation_loss = 0
    accuracy = 0
    with torch.no_grad(): #Desactivate autograd engine (reduce memory usage and speed up computations)
        network.eval() #set the layers to evaluation mode(batchnorm and dropout)
        for X, Y in validloader:
            if train_on_gpu:
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
            if indicePred==indice : accuracy += 1
        
        return validation_loss/len(validloader), 100*accuracy/(len(validloader))

def test(network, testloader):
    accuracy = 0
    with torch.no_grad(): #Desactivate autograd engine (reduce memory usage and speed up computations)
        network.eval() #set the layers to evaluation mode(batchnorm and dropout)
        for X, Y in testloader:
            if train_on_gpu:
                X=X.cuda()
                Y=Y.cuda()
            out = network(X)

            predict = network(X)
            valuePred, indicePred = predict[0].max(0) #get the value and indice of the predicted output (Highest probability)
            _, indice = Y.max(1) #get the true indice so we could compare to the predicted one
            
            if indicePred==indice : accuracy += 1
        
        print (100*accuracy/(len(testloader)))

def log(epochs, epoch, trainL, validL, accuracy):
    print("Epoch: {}/{}.. ".format(epoch, epochs-1),
        "Training Loss: {:.4f}.. ".format(trainL),
        "Validation Loss: {:.4f}.. ".format(validL),
        "Validation Accuracy: {:.3f}%".format(accuracy))

def show(trainL, validL):
    bestVali = min(validL)
    bestEpoch = validL.index(min(validL)) 
    plt.plot(trainL, label='Training loss')
    plt.plot(validL, label='Validation Loss')
    plt.title('Best validation perfomance is ' + str(bestVali) + ' at ' + str(bestEpoch))
    plt.legend(frameon=False)
    print('Best validation perfomance is ' + str(bestVali) + ' at ' + str(bestEpoch))
    plt.show()
