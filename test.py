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

# data = pd.read_csv('data/data.csv')

# print(data.head())
# corr.matrix(data.corr()['CLASS':'NSP'])
# print(data.shape)

# from sklearn.model_selection import train_test_split
# dataX=data.iloc[:,:-1]
# dataY=data.iloc[:,-1]

# X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, stratify=dataY, test_size=0.3)
# print(y_test.value_counts())

# print(data['NSP'].value_counts())

# from common.loader import CustomLoader as cl

# trainset = cl('data/CLASS/train.csv', 22)
# trainloader = DataLoader(dataset=trainset, batch_size=1)
# print(len(trainloader))


from common.nn_nsp import predict

# predict([120,0,0,0,73,0.5,])

# print('NSP 1')
# predict([137,1,0,0,20,2,0,0,5,0,0,74,86,160,1,0,126,128,130,23])
# print('NSP 2')
# predict([143,0,0,2,64,0.6,26,11.6,0,0,0,85,73,158,7,1,146,145,147,2])
# df[columns_keys[i]]

import pandas as pd

df = pd.read_csv('data/data_noenc.csv')
ck = {0: 'LB', 1: 'AC', 2: 'FM', 3: 'UC', 4: 'ASTV', 5: 'MSTV', 6: 'ALTV', 7: 'MLTV', 8: 'DL', 9: 'DS', 10: 'DP', 11: 'Width', 12: 'Min', 13: 'Max', 14: 'Nmax', 15: 'Nzeros', 16: 'Mode', 17: 'Mean', 18: 'Median', 19: 'Variance'}

maxMin=[]
for col in df.columns:
    maxi=df[col].max()
    mini=df[col].min()
    maxMin.append([mini,maxi])

print(maxMin)