#!/usr/bin/env python
# coding: utf-8

# 13)	Assignment on Hierarchical clustering. Apply it on Mall customers to form 5 clusters and display these clusters using scattergraph and also display dendrogram.

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# Loading Dataset

# In[33]:


df=pd.read_csv("C:/Users/rohit/OneDrive/Documents/6th sem/ML/Lab/ML_datasets/Mall_Customers.csv")
df.head()


# In[34]:


newdata=df.iloc[:,[3,4]].values


# Dendrogram

# In[35]:


dendrogram=sch.dendrogram(sch.linkage(newdata,method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()


# Model Building

# In[36]:


agg=AgglomerativeClustering(n_clusters=5,metric="euclidean",linkage="ward")


# Model Prediction

# In[37]:


y=agg.fit_predict(newdata)


# In[38]:


plt.scatter(newdata[y==0,0],newdata[y==0,1],s=100,c="red",label="Cluster 1")
plt.scatter(newdata[y==1,0],newdata[y==1,1],s=100,c="green",label="Cluster 2")
plt.scatter(newdata[y==2,0],newdata[y==2,1],s=100,c="blue",label="Cluster 3")
plt.scatter(newdata[y==3,0],newdata[y==3,1],s=100,c="orange",label="Cluster 4")
plt.scatter(newdata[y==4,0],newdata[y==4,1],s=100,c="purple",label="Cluster 5")
plt.title(" Cluster of Customer")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

