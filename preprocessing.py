import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing

def encoder(df1):
    df1 = df1.fillna(df1.mode().iloc[0])
    le = preprocessing.LabelEncoder()
    #print(df1)
    for i in range(len(df1.columns)):
        df1[df1.columns[i]] = le.fit_transform(df1[df1.columns[i]])
    """
    l = df1.isnull().sum(axis=0).tolist()
    avg = sum(l)/len(l)
    df1 = df1[np.isfinite(df1['Empathy'])]
    df1 = df1.fillna(0)
    """
    df1['Empathy'] = df1['Empathy'].map({0 : 0, 1 : 0, 2 : 0, 3 : 1, 4 : 1})
    """
    col = []
    i = 0
    while i < len(l):
        if l[i] > avg:
            col.append(i)
        i += 1
    df1 = df1.drop(col,axis = 0)
    """
    return df1
    
    
def removeIrrelaventFeatures(X,y):
    col = []
    corr = []
    for i in range(len (X.columns)):
        corr.append(X[X.columns[i]].corr(y[y.columns[0]]))
        if X[X.columns[i]].corr(y[y.columns[0]]) > -0.03 and X[X.columns[i]].corr(y[y.columns[0]]) < 0.01:
            col.append(i)
            #print(X.columns[i],X[X.columns[i]].corr(y[y.columns[0]]))
    X = X.drop(X.columns[col],axis = 1)
    return X



#helper function for the next function
def findCorelated(df):
    corrMatrix=df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > 0.6].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            #print(perfect_corr)
            result.append(perfect_corr)
        perfect_corr = corrMatrix[col][corrMatrix[col] < -0.6].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            #print(perfect_corr)
            result.append(perfect_corr)
    return result


def removeRedundantFeatues(X):
    corelated = findCorelated(X)
    for l in corelated:
        i = 0
        while i in range(len(l)-1):
            if l[i] in list(X.columns.values):
                X = X.drop(columns = [l[i]])
            i += 1
    return X


def Normalization(X):
    for i in range(len (X.columns)):
        X[X.columns[i]] -= X[X.columns[i]].mean()
        X[X.columns[i]] = X[X.columns[i]]/X[X.columns[i]].var()
    return X