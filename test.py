import pandas as pd
from pre_proc import corr

from torch.utils.data import DataLoader

# data = pd.read_csv('dataset.csv', sep='\t')

# # print(data.head())

# # corr.matrix(data.corr()['CLASS':'NSP'])

# # print(data.corr()['CLASS'])

# corr_class = data.corr()['CLASS']

# corr_class=corr_class.drop(['CLASS'])

# L = [p for p in corr_class.index if corr_class[p]>0]


# for i in L:
#     print(corr_class[i])

# data = pd.read_excel('data/CTG.xls', 'Raw data', skiprows=1)

data = pd.read_csv('data/data.csv')

print(data.head())
# corr.matrix(data.corr()['CLASS':'NSP'])
print(data.shape)

# from sklearn.model_selection import train_test_split
# dataX=data.iloc[:,:-1]
# dataY=data.iloc[:,-1]

# X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, stratify=dataY, test_size=0.3)
# print(y_test.value_counts())

# print(data['NSP'].value_counts())

from common.loader import CustomLoader as cl

trainset = cl('data/CLASS/train.csv', 22)
trainloader = DataLoader(dataset=trainset, batch_size=1)
print(len(trainloader))