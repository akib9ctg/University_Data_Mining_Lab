# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:41:20 2019

@author: ACER
"""

import pandas as pd
import numpy as np
dataset=pd.read_csv('Data.csv')

x=dataset.iloc[:,[1]]
y=dataset.iloc[:,[2]]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3.0,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)



import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Exp Train')
plt.xlabel('Salary')
plt.ylabel('Exp')
plt.show()


plt.scatter(x_test,y_pred,color='red')
plt.plot(x_test,regressor.predict(x_test),color='green')
plt.title('Salary Vs Exp Test')
plt.xlabel('Salary')
plt.ylabel('Exp')
plt.show()

#print(np.int(np.round(regressor.predict(50000))))
