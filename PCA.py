#!/usr/bin/env python
# coding: utf-8

# 14)	Assignment on Dimensionality Reduction:  Apply Principal component Analyses(PCA) on Iris dataset to reduce its dimensionality  into 3 principal componct Display data before and after reduction Scatter graph.

# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Loading Datset

# In[5]:


df = pd.read_csv("C:/Users/rohit/OneDrive/Documents/6th sem/ML/Lab/ML_datasets/iris.csv")
df.head()


# In[7]:


features = ['sepal_length','sepal_width','petal_length','petal_width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['species']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


# Spliting X and Y

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[10]:


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])


# In[11]:


finalDf = pd.concat([principalDf, df[['species']]], axis = 1)
finalDf.head()


# In[13]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['setosa', 'versicolor', 'virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['species'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[14]:


pca.explained_variance_ratio_


# In[15]:


#using original data
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[16]:


predictions


# In[17]:


accuracy_score(y_test, predictions)


# In[18]:


# Separating out the features
x = finalDf.drop(["species"], axis = 1)
x = StandardScaler().fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[19]:


#using PCA
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[20]:


predictions


# In[21]:


accuracy_score(y_test, predictions)

