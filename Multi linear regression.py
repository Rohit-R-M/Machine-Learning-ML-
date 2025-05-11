#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score


# Loading dataset

# In[6]:


df = pd.read_csv('C:/Users/3yearb1/Desktop/ML_datasets/USA_Housing.csv')


# In[7]:


display(df.head())


# In[8]:


sns.pairplot(df)


# Preparing X and Y

# In[11]:


X=df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y=df['Price']


# Splitting X and Y into training and testing sets

# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Building model

# In[13]:


model = LinearRegression()


# Model Training

# In[ ]:


model.fit(X_train, y_train)


# Model Testing

# In[ ]:


y_pred=model.predict(X_test)


# Calculating MSE and Accuracy

# In[16]:


mse =mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error:{mse}")
print(f"R-squared score: {r2}")

