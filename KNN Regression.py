#!/usr/bin/env python
# coding: utf-8

# 10)	Assignment on Regression using KNN. Build an application where it can predict Salary based on year of experience using KNN (use salary dataset from Kaggle). Display MSE.

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


# Loading Dataset

# In[15]:


df=pd.read_csv("C:/Users/rohit/OneDrive/Documents/6th sem/ML/Lab/ML_datasets/Salary_dataset.csv")
df.head()


# In[16]:


df.describe()


# Preparing X and Y

# In[17]:


X=df[['YearsExperience']]
Y=df['Salary']


# Spliting X and Y

# In[33]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=43)


# In[28]:


plt.scatter(X_train, Y_train, color='red')
plt.title('Salary VS Experience (Training Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Building model

# In[29]:


model=neighbors.KNeighborsRegressor(n_neighbors=3)


# Model Training

# In[21]:


model.fit(X_train,Y_train)


# Model Testing

# In[22]:


y_pred=model.predict(X_test)
y_pred


# Calculate MSE

# In[31]:


error=sqrt(mean_squared_error(Y_test,y_pred))
error

