# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 00:21:20 2019

@author: Mohammad Kafil Uddin
"""

import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [2,3,4]].values

from sklearn.cluster import KMeans
WCSS=[]     #within cluster sum of square | summation of distance from centroids to all the points within the cluster
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,init='k-means++',random_state=42) #init = initialization of centroids
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_) #inertia is to give wcss value of each and every cluster
print(WCSS)
plt.plot(range(1,11),WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=6,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(X[y_kmeans==0,0], X[y_kmeans==0,1], X[y_kmeans==0,2], s=100,c='red',label='cluster 1')
ax.scatter3D(X[y_kmeans==1,0], X[y_kmeans==1,1], X[y_kmeans==1,2], s=100,c='blue',label='cluster 2')
ax.scatter3D(X[y_kmeans==2,0], X[y_kmeans==2,1], X[y_kmeans==2,2], s=100,c='green',label='cluster 3')
ax.scatter3D(X[y_kmeans==3,0], X[y_kmeans==3,1], X[y_kmeans==3,2], s=100,c='cyan',label='cluster 4')
ax.scatter3D(X[y_kmeans==4,0], X[y_kmeans==4,1], X[y_kmeans==4,2], s=100,c='magenta',label='cluster 5')
ax.scatter3D(X[y_kmeans==5,0], X[y_kmeans==5,1], X[y_kmeans==5,2], s=100,c='yellow',label='cluster 6')
ax.scatter3D(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=300,c='black',label='centroids')

