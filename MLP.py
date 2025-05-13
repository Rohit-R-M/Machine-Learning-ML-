#!/usr/bin/env python
# coding: utf-8

# 9)	Assignment on Multiclassification using MLP (Multilayer Perception). Build an application to classify given iris flower Specie using MLP (Use Iris data set from Kaggle/ sklean). Display Accuracy score, classification report and confusion matrix.

# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier


# Load Dataset

# In[4]:


df=pd.read_csv("C:/Users/rohit/OneDrive/Documents/6th sem/ML/Lab/ML_datasets/iris.csv")
df.head()


# Preparing X and Y

# In[8]:


X=df[['sepal_length','sepal_width','petal_length','petal_width']]
Y=df['species']


# In[12]:


Y.unique() 


# In[18]:


pre=preprocessing.LabelEncoder()
Y=pre.fit_transform(Y)
Y


# Splitting X and Y

# In[19]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=43)
Y_test


# Model Building

# In[51]:


mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000,random_state=1)


# Model Training

# In[52]:


mlp.fit(X_train,Y_train)


# Model Testing

# In[47]:


y_pred=mlp.predict(X_test)
y_pred


# Calculating Confusion matrix,Accuracy Score,Classification report

# In[48]:


cm=confusion_matrix(Y_test,y_pred)
cr=classification_report(Y_test,y_pred)
accuracy=accuracy_score(Y_test,y_pred)
print(cr)
print(cm)
print("Accuracy Score: ",accuracy)

