# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 22:20:21 2019

@author: AKIB 
"""
import numpy as np
import pandas as pd

dataset=pd.read_csv('data.csv')

x=dataset.iloc[:,[1,2,3]].values
y=dataset.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score
rms=np.sqrt(mean_squared_error(y_test,y_pred))
print(rms)
r2_score=r2_score(y_test,y_pred)
print(r2_score)