import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre_proc import corr
# data = pd.read_excel('CTG.xls', 'Data', skiprows=1)


# print(data.head())
# print(data.columns)
def del_unnecessary(data): #data is the original csv
    '''
        Delete empty columns from the dataset and return two datafram:
        One with the encoded output and one without.
    '''
    
    data_noenc = data.copy()
    data_noenc.drop(['LBE','b', 'e','DR','Tendency','A','B','C','D','E','AD','DE','LD','FS','SUSP'], axis=1, inplace=True)
    data.drop(['LBE','b', 'e', 'DR','Tendency'], axis=1, inplace=True)
    data.to_csv("data/data_enc.csv", sep=",", index=False)
    data_noenc.to_csv("data/data_noenc.csv", sep=",", index=False)
    return data,data_noenc

def postitive_corr(data): #data here is the "data_noenc" returned from del_unnecessarry
    '''
        Save the attributes that have positive correlation with 
        our classes(CLASS and NSP)
    '''


    #####  The "CLASS" class  #####
    corr_class = data.corr()['CLASS']
    corr_class=corr_class.drop(['CLASS'])
    L = [p for p in corr_class.index if corr_class[p]>0]
    with open('data/corr_class.txt', 'w') as output:
        output.write(str(L))


    #####  The "NSP" class  #####
    corr_nsp = data.corr()['NSP']
    corr_nsp=corr_nsp.drop(['NSP'])
    L = [p for p in corr_nsp.index if corr_nsp[p]>0]
    with open('data/corr_nsp.txt', 'w') as output:
        output.write(str(L))

def normalise(data):
    data['LB'] = (data['LB'] - data['LB'].min()) / (data['LB'].max() - data['LB'].min())
    data['AC'] = (data['AC'] - data['AC'].min()) / (data['AC'].max() - data['AC'].min())
    data['FM'] = (data['FM'] - data['FM'].min()) / (data['FM'].max() - data['FM'].min())
    data['UC'] = (data['UC'] - data['UC'].min()) / (data['UC'].max() - data['UC'].min())
    data['ASTV'] = (data['ASTV'] - data['ASTV'].min()) / (data['ASTV'].max() - data['ASTV'].min())
    data['MSTV'] = (data['MSTV'] - data['MSTV'].min()) / (data['MSTV'].max() - data['MSTV'].min())
    data['ALTV'] = (data['ALTV'] - data['ALTV'].min()) / (data['ALTV'].max() - data['ALTV'].min())
    data['MLTV'] = (data['MLTV'] - data['MLTV'].min()) / (data['MLTV'].max() - data['MLTV'].min())
    data['DL'] = (data['DL'] - data['DL'].min()) / (data['DL'].max() - data['DL'].min())
    data['DS'] = (data['DS'] - data['DS'].min()) / (data['DS'].max() - data['DS'].min())
    data['DP'] = (data['DP'] - data['DP'].min()) / (data['DP'].max() - data['DP'].min())
    data['Width'] = (data['Width'] - data['Width'].min()) / (data['Width'].max() - data['Width'].min())
    data['Min'] = (data['Min'] - data['Min'].min()) / (data['Min'].max() - data['Min'].min())
    data['Max'] = (data['Max'] - data['Max'].min()) / (data['Max'].max() - data['Max'].min())
    data['Nmax'] = (data['Nmax'] - data['Nmax'].min()) / (data['Nmax'].max() - data['Nmax'].min())
    data['Nzeros'] = (data['Nzeros'] - data['Nzeros'].min()) / (data['Nzeros'].max() - data['Nzeros'].min())
    data['Mode'] = (data['Mode'] - data['Mode'].min()) / (data['Mode'].max() - data['Mode'].min())
    data['Mean'] = (data['Mean'] - data['Mean'].min()) / (data['Mean'].max() - data['Mean'].min())
    data['Median'] = (data['Median'] - data['Median'].min()) / (data['Median'].max() - data['Median'].min())
    data['Variance'] = (data['Variance'] - data['Variance'].min()) / (data['Variance'].max() - data['Variance'].min())
    # data['Tendency'] = (data['Tendency'] - data['Tendency'].min()) / (data['Tendency'].max() - data['Tendency'].min())


    # df["A"] = df["A"] / df["A"].max()

    return data

def normaliseRow(row):
    """
        Normalise one row only(for prediction)
    """
    # df = pd.read_csv('data/data_noenc.csv')
    # columns_keys={i:col for i,col in enumerate(df.columns[:20])}
    # maxMin=[]
    # for col in df.columns: #del the last 2
    #     maxi=df[col].max()
    #     mini=df[col].min()
    #     maxMin.append([mini,maxi])
    columns_keys = {0: 'LB', 1: 'AC', 2: 'FM', 3: 'UC', 4: 'ASTV', 5: 'MSTV', 6: 'ALTV', 7: 'MLTV', 8: 'DL', 9: 'DS', 10: 'DP', 11: 'Width', 12: 'Min', 13: 'Max', 14: 'Nmax', 15: 'Nzeros', 16: 'Mode', 17: 'Mean', 18: 'Median', 19: 'Variance'}
    maxMin=[[106, 160], [0, 26], [0, 564], [0, 23], [12, 87], [0.2, 7.0], [0, 91], [0.0, 50.7], [0, 16], [0, 1], [0, 4], [3, 180], [50, 159], [122, 238], [0, 18], [0, 10], [60, 187], [73, 182], [77, 186], [0, 269]]
    for i in range(len(row)):
        row[i] = (row[i] - maxMin[i][0]) / (maxMin[i][1] - maxMin[i][0])

    return row

def split(data):
    '''
        Split the data into 70% training 15% validation and 15% testing.
        Same percentage of classes is present in all three.
    '''
    # data = normalise(data)
    
    dataX=data.iloc[:,:-1]
    dataY=data.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3, random_state=0, stratify=dataY)


    data_train = pd.concat([X_train,y_train],axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)

    data_vali = pd.concat([X_train,y_train],axis=1)
    data_test = pd.concat([X_test,y_test],axis=1)

    return data_train, data_vali, data_test

def oneHotEncode_NSP(data_train, data_vali, data_test):
    '''
        Onehote encode the NSP class
    '''

    onehot = pd.get_dummies(data_train['NSP'], prefix = 'NSP')
    data_train = pd.concat([data_train,onehot],axis=1)
    onehot2 = pd.get_dummies(data_vali['NSP'], prefix = 'NSP')
    data_vali = pd.concat([data_vali,onehot2],axis=1)
    onehot3 = pd.get_dummies(data_test['NSP'], prefix = 'NSP')
    data_test = pd.concat([data_test,onehot3],axis=1)

    return data_train, data_vali, data_test

def run():
    data, data_noenc = del_unnecessary(normalise(pd.read_csv('data/data.csv')))



    postitive_corr(data_noenc) # Save in a txt file the positive corelations for each class


    del data['NSP']
    data_train, data_vali, data_test = split(data)

    del data_train['CLASS']
    del data_vali['CLASS']
    del data_test['CLASS']

    data_train.to_csv("data/CLASS/train.csv", sep=",", index=False)
    data_vali.to_csv("data/CLASS/vali.csv", sep=",", index=False)
    data_test.to_csv("data/CLASS/test.csv", sep=",", index=False)



    del data_noenc['CLASS']

    data_train, data_vali, data_test = split(data_noenc)
    data_train, data_vali, data_test = oneHotEncode_NSP(data_train, data_vali, data_test)

    del data_train['NSP']
    del data_vali['NSP']
    del data_test['NSP']

    data_train.to_csv("data/NSP/train.csv", sep=",", index=False)
    data_vali.to_csv("data/NSP/vali.csv", sep=",", index=False)
    data_test.to_csv("data/NSP/test.csv", sep=",", index=False)


