# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 20:31:27 2019

@author: AKib
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


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x_train);
regressor.fit(x_poly,y_train)
y_pred=regressor.predict(x_poly)



import matplotlib.pyplot as plt
plt.scatter(x_train,y_pred,color='red')
plt.plot(x_train,y_pred,color='Green')
plt.title('Salary Vs Experience (Train)')
plt.xlabel('Salary')
plt.ylabel('Experience')
plt.show()


#Finding mean Squared Error
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
r2 = r2_score(y_train,y_pred)
print(rmse)
print(r2)


