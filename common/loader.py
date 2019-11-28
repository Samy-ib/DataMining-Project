import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder

import torch.nn.functional as F
from torch import nn, optim

class CustomLoader(Dataset):
    def __init__(self, chemin, j):
        """
            Receives a Pandas dataframe of the dataset that sould be normalized.
            It should only contains our one class.
            The variable "j" is the column number in the dataframe at which the
            class start.
        """
        data=pd.read_csv(chemin)
        x = data.iloc[:, :j]
        y = data.iloc[:, j:] 
        self.len = x.shape[0]
        self.x_data = torch.tensor(x.values).float()
        self.y_data = torch.tensor(y.values).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len