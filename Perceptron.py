#!/usr/bin/env python
# coding: utf-8

# 8)	Assignment on Binary classification using Percoptron. Implement Perception model. Use this model to classify a patient is having cancer or not. (use Breast cancer dataset from sklearn), Display Accuracy score, classification Report and Confusion matrix

# In[1]:


from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[2]:


bc= load_breast_cancer()


# In[3]:


X=bc.data
Y=bc.target


# In[4]:


data=pd.DataFrame(bc.data,columns=bc.feature_names)
data['class']=bc.target
data.head()


# In[5]:


X=data.drop('class',axis=1)
Y=data['class']


# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y)


# In[15]:


X_train=X_train.values
X_test=X_test.values
Y_train=Y_train.values
Y_test = Y_test.values


# In[21]:


# Perceptron model function
def model(w, b, x):
    return 1 if (np.dot(w, x) >= b) else 0

# Predict function for all samples
def predict(w, b, X):
    Y = []
    for x in X:
        result = model(w, b, x)
        Y.append(result)
    return np.array(Y)

# Training function
def fit(X, Y, epochs=1, lr=1):
    w = np.ones(X.shape[1])
    b = 0

    accuracy = {}
    max_accuracy = 0

    wt_matrix = []

    for i in range(epochs):
        for x, y in zip(X, Y):
            y_pred = model(w, b, x)
            if y == 1 and y_pred == 0:
                w = w + lr * x
                b = b - lr * 1
            elif y == 0 and y_pred == 1:
                w = w - lr * x
                b = b + lr * 1

        wt_matrix.append(w.copy())
        
        current_accuracy = accuracy_score(predict(w, b, X), Y)
        accuracy[i] = current_accuracy

        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            chkptw = w.copy()
            chkptb = b


    print(max_accuracy)

    plt.plot(accuracy.values())
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Epochs")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()
    
    return np.array(wt_matrix), chkptw, chkptb


# In[22]:


wt_matrix, w, b = fit(X_train, Y_train, 10000, 0.5)
print(w)


# In[18]:


Y_pred_test=predict(w,b,X_test)
print(accuracy_score(Y_pred_test,Y_test))

