#!/usr/bin/env python
# coding: utf-8

# 11)	Assignment on Classification using KNN. Build an application to classify an iris flower into its specie using KNN (Iris dataset from Skleam). Display Accuracy score, classification Report & Confusion Matrix).

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.datasets import load_iris
from sklearn import neighbors


# Loading Dataset

# In[26]:


iris=load_iris()


# Preparing X and Y

# In[27]:


X=iris.data
Y=iris.target


# Spliting X and Y

# In[21]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


# Model Building

# In[22]:


model = neighbors.KNeighborsClassifier(n_neighbors=3)


# Model Training

# In[23]:


model.fit(X_train,Y_train)


# Model Testing

# In[24]:


y_pred=model.predict(X_test)
y_pred


# In[25]:


accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy Score: ",accuracy)

print("\nClassification Report:")
print(classification_report(Y_test, y_pred,))

cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", cm)

