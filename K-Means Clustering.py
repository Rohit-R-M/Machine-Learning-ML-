#!/usr/bin/env python
# coding: utf-8

# 12)	Assignment on K-mean clustering. Apply K-mean clustering on Income data set to form 3 Clusters and display there clasters using scatter graph.

# In[124]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*KMeans is known to have a memory leak.*")


# Loading Dataset

# In[125]:


df=pd.read_csv("C:/Users/rohit/OneDrive/Documents/6th sem/ML/Lab/ML_datasets/income.csv")
df.head()


# In[126]:


plt.scatter(df.Age,df['Income($)'],color="red")
plt.title("Age vs Income")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()


# Preparing X 

# In[127]:


X=df[['Age','Income($)']]


# Building Model

# In[128]:


model=KMeans(n_clusters=3,n_init=10)
y_pred=model.fit_predict(X)
y_pred


# Add cluster labels to dataframe

# In[129]:


df['cluster']=y_pred
df.head()


# Plot Scaled Data Cluster

# In[130]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1["Income($)"],color="red")
plt.scatter(df2.Age,df2["Income($)"],color="green")
plt.scatter(df3.Age,df3["Income($)"],color="blue")

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='orange',marker='*',label='centriod')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()


# Preprocessing using MinMaxScaler

# In[131]:


scaler=MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[["Income($)"]])

scaler.fit(df[["Age"]])
df["Age"]=scaler.transform(df[["Age"]])


# In[132]:


df.head()


# In[133]:


plt.scatter(df.Age,df["Income($)"])


# In[134]:


model=KMeans(n_clusters=3,n_init=10)
y_pred=model.fit_predict(df[['Age','Income($)']])
y_pred


# In[135]:


df['cluster']=y_pred
df.head()


# In[136]:


model.cluster_centers_


# In[137]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1["Income($)"],color="red")
plt.scatter(df2.Age,df2["Income($)"],color="green")
plt.scatter(df3.Age,df3["Income($)"],color="blue")

plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],color='orange',marker='*',label='centriod')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()


# Calculate Sum of Squared errors

# In[150]:


sse=[]
k_range=range(1,10)
for k in k_range:
    km=KMeans(n_clusters=k,n_init=10)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)
sse


# In[151]:


plt.plot(k_range,sse)
plt.xlabel('Numbers of K')
plt.ylabel('Sum of squared error')
plt.show()

