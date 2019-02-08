# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:54:43 2018

@author: ACER
"""
import numpy as np
import pandas as pd
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
#Can also be written as x=dataset.iloc[:,[0,1,2]].values

y=dataset.iloc[:,3].values
print(x)
print(y)
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,-1:])
x[:,-1:]=imputer.transform(x[:,-1:])