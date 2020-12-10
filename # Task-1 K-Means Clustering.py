#!/usr/bin/env python
# coding: utf-8

# ### Name : Nida Bijapure
# 
# ### Task-1 : Prediction using Unsupervised ML ( K-Means Clustering)
# 
# ### From the given 'Iris' dataset we have to predict the optimum number of clusters and represent it visually.

# ## Import the relevant libraries

# In[2]:


# Importing the relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# ## Load The Data

# In[3]:


# Load the iris dataset

iris_data = pd.read_csv('Iris.csv')
iris_data.head(5)


# In[4]:


# Dropping the unuseful columns from the dataset
iris_data.drop('Id',axis=1,inplace=True)
iris_data.head(5)


# In[7]:


# Descritive Analysis of data

print('Dimension of dataset: {}'.format(iris_data.shape))
print('Descriptives:\n {}'.format(iris_data.describe()))
print('Correlation:\n {}'.format(iris_data.corr()))
print('Count of Species:\n {}'.format(iris_data['Species'].value_counts()))


# ## How to Choose the Number of Clusters (K)  

# Using the code below we will find the <i> <b>Within Clusters Sum of Squares(WCSS)</b></i> for clustering solutions with 1 to 10 clusters (you can try with more if you wish).
# 
# Find the most suitable solutions, run them and compare the results.

# In[9]:


# Finding the optimum number of clusters for k-means classification

x = iris_data.iloc[:,0:4].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# ### Visualizing the Elbow Graph 

# In[10]:


# Plotting the results onto a line graph,allowing us to observe 'The elbow'

sns.set(style = "darkgrid")
plt.plot(range(1, 11), wcss, marker ='o', color = 'blue')
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# <b>You can clearly see from the above graph,the elbow shape occurs at point 3. That is from point 3 the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# So therefore in this case the optimum number of clusters is 3.</b>

# In[11]:


# Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)


# In[14]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 50, c = 'green', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 50, c = 'purple', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 50, c = 'red', label = 'Centroids')

plt.legend(loc='best',fontsize='medium')


# In[ ]:




