#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


#importing the dataset with pandas
dataset = pd.read_csv('Iris.csv')


# In[6]:


dataset


# In[10]:


X = dataset.iloc[:,[1,2,3,4]].values


# In[11]:


X


# In[13]:


#using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[14]:


#applying kmeans to the dataset with value of k=3 in this case
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# In[15]:


y_kmeans


# In[32]:


#visualising the clusters wrt sepals
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 30, c = 'red', label = 'Iris-virginica')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 30, c = 'blue', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 30, c = 'green', label = 'Iris-versicolor')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 70, c = 'yellow', label = 'Centroid')
plt.title('Clusters of Species')
plt.xlabel('Sepal Length in cm')
plt.ylabel('Sepal Width in cm')
plt.legend()
plt.show()


# In[31]:


#visualising the clusters wrt petals
plt.scatter(X[y_kmeans == 0, 2], X[y_kmeans == 0, 3], s = 30, c = 'red', label = 'Iris-virginica')
plt.scatter(X[y_kmeans == 1, 2], X[y_kmeans == 1, 3], s = 30, c = 'blue', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 2, 2], X[y_kmeans == 2, 3], s = 30, c = 'green', label = 'Iris-versicolor')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s = 70, c = 'yellow', label = 'Centroid')
plt.title('Clusters of Species')
plt.xlabel('Petal Length in cm')
plt.ylabel('Petal Width in cm')
plt.legend()
plt.show()


# In[ ]:




