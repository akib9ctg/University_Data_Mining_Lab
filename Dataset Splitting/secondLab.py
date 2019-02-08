# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 00:55:04 2018

@author: ACER
"""
import numpy as np
import pandas as pd

dataset=pd.read_csv('Data.csv')

x=dataset.iloc[:,:]
# from sklearn.cross_validation import train_test_split 
#Depricated

from sklearn.model_selection import train_test_split


x_train,x_test=train_test_split(x,test_size=0.2,random_state=0)
#Here cause of random_state_value, the x_train, x_test will be fixed
#But x_train,x_test=train_test_split(x,test_size=0.2) 
#using this line x_test, x_train value will be random

print(x_train)
print(x_test)