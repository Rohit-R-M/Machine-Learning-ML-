#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report


# Load Dataset

# In[59]:


df=pd.read_csv("C:/Users/3yearb1/Desktop/ML_datasets/PlayTennis.csv")


# In[60]:


display(df)


# In[47]:


print(df.describe())


# In[48]:


outlook = df["Outlook"].str.get_dummies(" ")
print(outlook)


# In[49]:


temp = df["Temperature"].str.get_dummies(" ")
print(temp)


# In[50]:


hum = df["Humidity"].str.get_dummies(" ")
print(hum)


# In[51]:


wind = df["Wind"].str.get_dummies(" ")
print(wind)


# In[52]:


playtennis = df["Play Tennis"].str.get_dummies(" ")
print(playtennis)


# In[61]:


df.drop(['Outlook','Temperature','Humidity','Wind','Play Tennis'],axis=1,inplace=True)


# In[62]:


df=pd.concat([outlook,temp,hum,wind,playtennis],axis=1)       


# In[66]:


display(df)


# Prepare X and Y

# In[67]:


x=df.drop(['Yes','No'],axis=1)
y=df[['Yes']]


# Split X & Y into training and testing dataset

# In[89]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# Building Model

# In[90]:


dt=DecisionTreeClassifier(criterion = 'entropy')


# Training Model

# In[91]:


dt.fit(X_train,y_train)


# Testing Model

# In[92]:


y_pred = dt.predict(X_test)


# In[93]:


print(y_pred)


# Calculating Confusion matrix and Classification report

# In[94]:


m=confusion_matrix(y_test,y_pred)


# In[95]:


m


# In[100]:


report = classification_report(y_test, y_pred)
display(report)


# In[ ]:




